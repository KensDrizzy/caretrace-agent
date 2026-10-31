"""Isolated reliability tests; real stdio MCP, no LLM, SMTP delivery or production DB."""
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pytest
from openpyxl import load_workbook
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.core.tool_schema import migrate_tool_schema
from app.models.entities import (UserAccount, ChatSession, PsychologicalReport, ToolJob, RiskCase,
                                 AlertRecord, ExcelRecord, ToolAuditRecord, DeadLetterRecord, AgentTraceEvent, now)
from app.services.mcp_client import MindBridgeMcpToolClient, McpToolError
from app.services.tool_queue import ToolQueueService, ToolQueueWorker
from app.services.tool_execution import execute_tool
from app.services.tool_errors import ToolExecutionError, DeliveryUnknown
from app.services.tools import ToolOrchestrationService


@pytest.fixture
def env(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.sqlite", excel_path=str(tmp_path/'ledger.xlsx'),
                        alert_email_delivery_mode='log', alert_email_rate_limit_per_minute=0,
                        smtp_host='', smtp_username='', smtp_password='', alert_email_to='', alert_email_from='',
                        tool_queue_retry_delay_seconds=1, tool_queue_lease_seconds=10)
    mysql_url = os.environ.get('TEST_TOOL_MYSQL_URL')
    if mysql_url:
        from sqlalchemy.engine import make_url
        assert make_url(mysql_url).database == 'caretrace_test_tools', 'Only disposable test database allowed'
        settings.database_url = mysql_url
        engine = create_engine(mysql_url, pool_pre_ping=True)
        Base.metadata.drop_all(engine)
    else:
        engine = create_engine(settings.database_url, connect_args={'check_same_thread': False, 'timeout': 15})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    with factory() as db:
        user = UserAccount(username='synthetic', display_name='Synthetic', password_hash='x')
        db.add(user); db.flush()
        session = ChatSession(public_id='synthetic', user_id=user.id, title='Synthetic')
        db.add(session); db.flush()
        for risk in ('HIGH','LOW','HIGH'):
            db.add(PsychologicalReport(user_id=user.id, session_id=session.id, content='Synthetic fixture', intent='RISK',
                emotion='ANXIETY', emotion_score=1, risk_level=risk, confidence=.9, summary='Synthetic'))
        db.commit()
    yield SimpleNamespace(settings=settings, engine=engine, factory=factory)
    if mysql_url:
        Base.metadata.drop_all(engine)
    engine.dispose()


def test_real_stdio_worker_pipeline_and_receipts(env):
    with env.factory() as db:
        jobs = ToolQueueService(db, env.settings).enqueue_report(1, 'HIGH', trace_id='audit-trace')
        ids = {j.kind: j.id for j in jobs}
        assert len(ToolQueueService(db, env.settings).enqueue_report(1, 'HIGH')) == 3
        assert db.query(ToolJob).count() == 3
    worker = ToolQueueWorker(env.settings, env.factory)
    try:
        for kind in ('CASE_CREATE','ALERT_SEND','EXCEL_REPORT'):
            token = worker._claim_job(ids[kind])
            assert token
            worker._run_job(ids[kind], token)
        with env.factory() as db:
            assert db.get(ToolJob, ids['CASE_CREATE']).status == 'SUCCESS'
            assert db.get(ToolJob, ids['EXCEL_REPORT']).status == 'SUCCESS'
            alert_job = db.get(ToolJob, ids['ALERT_SEND'])
            assert alert_job.status == 'SIMULATED', alert_job.last_error
            assert db.query(AlertRecord).one().status == 'SIMULATED'
            assert db.query(RiskCase).one().status == 'OPEN'
            assert db.query(ToolAuditRecord).count() == 3
            assert db.query(AgentTraceEvent).filter_by(event_type='TOOL_EXECUTION_COMPLETED').count() == 2
        client = MindBridgeMcpToolClient(env.settings)
        asyncio.run(client.handle_report(1, 'HIGH'))
        with env.factory() as db:
            assert db.query(ExcelRecord).count() == 1
            assert db.query(AlertRecord).count() == 1
        workbook = load_workbook(env.settings.excel_path)
        assert sum(row[0] == 1 for row in workbook.active.iter_rows(values_only=True)) == 1
        workbook.close()
    finally:
        worker.stop()


def test_real_mcp_errors_policy_and_simulation_does_not_suppress_smtp(env):
    async def call():
        client = MindBridgeMcpToolClient(env.settings)
        async with client._session() as session:
            missing = await session.call_tool('mindbridge_excel_report', arguments={'report_id':9999})
            assert missing.isError
            with pytest.raises(McpToolError) as denied:
                await client._call_tool(session,'mindbridge_alert_notify',{'report_id':2})
            assert denied.value.code == 'POLICY_DENIED'
            assert not denied.value.retryable
        await client.handle_report(1,'HIGH')
        env.settings.alert_email_delivery_mode = 'smtp'
        with pytest.raises(McpToolError) as error:
            await client.handle_report(1,'HIGH')
        assert error.value.code == 'DELIVERY_FAILED'
        assert not error.value.retryable
    asyncio.run(call())
    with env.factory() as db:
        assert {a.status for a in db.query(AlertRecord)} == {'SIMULATED','FAILED'}
        assert db.query(ToolAuditRecord).filter_by(status='BLOCKED').count() >= 2


def test_two_workers_claim_once_and_stale_owner_cannot_complete(env):
    with env.factory() as db:
        job_id = ToolQueueService(db,env.settings).enqueue_report(2,'LOW')[0].id
    workers = [ToolQueueWorker(env.settings,env.factory) for _ in range(2)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            tokens = list(pool.map(lambda worker:worker._claim_job(job_id), workers))
        assert sum(token is not None for token in tokens) == 1
        with env.factory() as db:
            job=db.get(ToolJob,job_id)
            stale=SimpleNamespace(**{name:getattr(job,name) for name in ('id','lease_token','trace_id','kind','report_id','attempts','idempotency_key')})
            job.lease_until=now()-timedelta(seconds=5); db.commit()
        workers[0]._recover_running_jobs()
        with env.factory() as db:
            assert db.get(ToolJob,job_id).status=='RETRY_WAIT'
            assert not workers[0]._transition(db,stale,'SUCCESS')
    finally:
        for worker in workers: worker.stop()


def test_live_lease_is_not_reset_on_startup(env):
    with env.factory() as db:
        job_id=ToolQueueService(db,env.settings).enqueue_report(2,'LOW')[0].id
    worker=ToolQueueWorker(env.settings,env.factory)
    try:
        token=worker._claim_job(job_id)
        worker._recover_running_jobs()
        with env.factory() as db:
            assert db.get(ToolJob,job_id).lease_token==token
            assert db.get(ToolJob,job_id).status=='RUNNING'
    finally: worker.stop()


def test_dead_dependency_is_terminal_and_no_mcp_execution(env):
    class NeverCalled:
        def __init__(self,*args): raise AssertionError('MCP should not execute')
    with env.factory() as db:
        jobs=ToolQueueService(db,env.settings).enqueue_report(1,'HIGH')
        case=next(j for j in jobs if j.kind=='CASE_CREATE'); case.status='DEAD'
        alert_id=next(j.id for j in jobs if j.kind=='ALERT_SEND'); db.commit()
    worker=ToolQueueWorker(env.settings,env.factory,NeverCalled)
    try:
        worker._run_job(alert_id,worker._claim_job(alert_id))
        with env.factory() as db:
            assert db.get(ToolJob,alert_id).status=='DEAD'
            assert 'DEPENDENCY_FAILED' in db.get(ToolJob,alert_id).last_error
            assert db.query(DeadLetterRecord).count()==1
    finally: worker.stop()


def test_retry_then_dead_and_admin_replay(env):
    class FailingClient:
        def __init__(self,*args): pass
        async def execute_job(self,*args): raise ToolExecutionError('temporary',code='TEMPORARY')
    with env.factory() as db:
        job=ToolQueueService(db,env.settings).enqueue_report(2,'LOW')[0]; job.max_attempts=2; db.commit(); job_id=job.id
    worker=ToolQueueWorker(env.settings,env.factory,FailingClient)
    try:
        worker._run_job(job_id,worker._claim_job(job_id))
        with env.factory() as db:
            job=db.get(ToolJob,job_id); assert job.status=='RETRY_WAIT'
            job.run_after=now()-timedelta(seconds=1); db.commit()
        worker._run_job(job_id,worker._claim_job(job_id))
        with env.factory() as db:
            assert db.get(ToolJob,job_id).status=='DEAD'
            ToolQueueService(db,env.settings).replay(job_id,'admin','configuration corrected')
            assert db.get(ToolJob,job_id).status=='PENDING'
            assert db.query(DeadLetterRecord).count()==1
    finally: worker.stop()


def test_report_and_jobs_rollback_together(env,monkeypatch):
    from app.agents.harness import MindBridgeAgentHarness
    from app.core.enums import IntentType, RiskLevel, EmotionLabel
    with env.factory() as db:
        harness=object.__new__(MindBridgeAgentHarness); harness.db=db; harness.settings=env.settings
        original=ToolQueueService._find_or_create
        def fail_after_first(self,kind,*args,**kwargs):
            if kind=='CASE_CREATE': raise RuntimeError('injected enqueue failure')
            return original(self,kind,*args,**kwargs)
        monkeypatch.setattr(ToolQueueService,'_find_or_create',fail_after_first)
        run=SimpleNamespace(requires_report=True,trace_id='atomic',intent=IntentType.RISK,
            assessment=SimpleNamespace(emotion=EmotionLabel.ANXIETY,emotion_score=1,risk=RiskLevel.HIGH,confidence=.9,summary='Synthetic'))
        with pytest.raises(RuntimeError): harness._create_report(db.get(UserAccount,1),db.get(ChatSession,1),'Synthetic',run)
        assert db.query(PsychologicalReport).count()==3
        assert db.query(ToolJob).count()==0


def test_excel_recovers_file_saved_before_db_commit(env,monkeypatch):
    with env.factory() as db:
        tools=ToolOrchestrationService(db,env.settings); report=db.get(PsychologicalReport,1)
        commit=db.commit
        monkeypatch.setattr(db,'commit',lambda: (_ for _ in ()).throw(RuntimeError('crash after file save')))
        with pytest.raises(RuntimeError): tools.write_excel(report)
        db.rollback(); monkeypatch.setattr(db,'commit',commit)
        tools.write_excel(report)
        workbook=load_workbook(env.settings.excel_path)
        assert sum(row[0]==1 for row in workbook.active.iter_rows(values_only=True))==1
        workbook.close()


def test_unknown_delivery_never_automatically_resends(env,monkeypatch):
    env.settings.alert_email_delivery_mode='smtp'; env.settings.smtp_host='synthetic.invalid'
    env.settings.alert_email_to='synthetic@example.invalid'; env.settings.alert_email_from='test@example.invalid'
    calls=[]
    def unknown(*args):
        calls.append(1)
        raise DeliveryUnknown('synthetic lost acknowledgement')
    monkeypatch.setattr(ToolOrchestrationService,'_send_alert_email',unknown)
    for _ in range(2):
        with env.factory() as db:
            with pytest.raises(ToolExecutionError) as error:
                execute_tool(db,env.settings,'RISK_ALERT',1)
            assert error.value.code=='DELIVERY_UNKNOWN'
            assert not error.value.retryable
    assert len(calls)==1


def test_sending_record_after_crash_becomes_unknown(env,monkeypatch):
    env.settings.alert_email_delivery_mode='smtp'; env.settings.smtp_host='synthetic.invalid'
    env.settings.alert_email_to='synthetic@example.invalid'; env.settings.alert_email_from='test@example.invalid'
    with env.factory() as db:
        tools=ToolOrchestrationService(db,env.settings)
        monkeypatch.setattr(tools,'_send_alert_email',lambda *args: (_ for _ in ()).throw(SystemExit('crash')))
        with pytest.raises(SystemExit): tools.notify(db.get(PsychologicalReport,1))
    with env.factory() as db:
        result=ToolOrchestrationService(db,env.settings).notify(db.get(PsychologicalReport,1))
        assert result.status=='UNKNOWN' and not result.retryable


def test_migration_idempotent_on_existing_schema(env):
    migrate_tool_schema(env.engine,env.settings)
    migrate_tool_schema(env.engine,env.settings)
    with env.factory() as db:
        assert db.query(PsychologicalReport).count()==3


def test_sse_disconnect_keeps_report_and_jobs(env,monkeypatch):
    import app.agents.harness as harness_module
    from app.agents.result import AgentRunResult
    from app.core.enums import IntentType, RiskLevel, EmotionLabel
    from app.harness.runner import InMemoryShortTermMemoryStore
    from app.schemas.dtos import ChatRequest
    from app.services.assessment import PsychologyAssessment
    from app.services.chat import ChatService
    run=AgentRunResult(intent=IntentType.RISK,risk_level=RiskLevel.HIGH,
        assessment=PsychologyAssessment(EmotionLabel.HIGH_RISK,4,RiskLevel.HIGH,.9,'Synthetic'),
        retrieved_knowledge=[],response_messages=[],steps=[],memory_brief='',final_response='Synthetic response',trace_id='disconnect')
    monkeypatch.setattr(harness_module,'create_agent_runtime',lambda *args:SimpleNamespace(run=lambda *args:run))
    monkeypatch.setattr(harness_module,'RedisShortTermMemoryStore',InMemoryShortTermMemoryStore)
    async def disconnect():
        with env.factory() as db:
            service=ChatService(db,env.settings)
            stream=service.stream_chat(db.get(UserAccount,1),ChatRequest(message='Synthetic',sessionId='synthetic'))
            first=await anext(stream)
            assert 'event: meta' in first
            await stream.aclose()
    asyncio.run(disconnect())
    with env.factory() as db:
        jobs=db.query(ToolJob).filter_by(trace_id='disconnect').all()
        assert len(jobs)==3
        assert db.query(PsychologicalReport).count()==4
        assert all(job.status=='PENDING' for job in jobs)
        assert db.query(AgentTraceEvent).filter_by(event_type='TOOL_EXECUTION_QUEUED').count()==3


def test_concurrent_submissions_have_unique_jobs(env):
    def submit(_):
        with env.factory() as db:
            return [j.id for j in ToolQueueService(db,env.settings).enqueue_report(1,'HIGH')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids=list(pool.map(submit,range(2)))
    assert ids[0]==ids[1]
    with env.factory() as db:
        assert db.query(ToolJob).count()==3


def test_rate_limit_deferral_does_not_exhaust_delivery_attempts(env):
    class Throttled:
        def __init__(self,*args): pass
        async def execute_job(self,*args):
            raise ToolExecutionError('throttled',code='RATE_LIMITED',retry_after=15)
    with env.factory() as db:
        job_id=ToolQueueService(db,env.settings).enqueue_report(2,'LOW')[0].id
    worker=ToolQueueWorker(env.settings,env.factory,Throttled)
    try:
        worker._run_job(job_id,worker._claim_job(job_id))
        with env.factory() as db:
            job=db.get(ToolJob,job_id)
            assert job.status=='RETRY_WAIT'
            assert job.attempts==0
            assert db.query(DeadLetterRecord).count()==0
    finally: worker.stop()


def test_redis_limit_shared_by_independent_clients_and_fail_closed(env,tmp_path):
    import shutil
    import subprocess
    import time
    import redis
    from app.services.tool_rate_limit import require_email_capacity
    executable=shutil.which('redis-server')
    if not executable:
        pytest.skip('redis-server not installed')
    import tempfile
    from pathlib import Path
    directory=tempfile.TemporaryDirectory(prefix='ct-redis-',dir='/tmp')
    socket=Path(directory.name)/'redis.sock'
    process=subprocess.Popen([executable,'--port','0','--unixsocket',str(socket),'--save','','--appendonly','no'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        env.settings.redis_url=f'unix://{socket}'
        env.settings.alert_email_rate_limit_per_minute=1
        client=redis.Redis.from_url(env.settings.redis_url)
        for _ in range(100):
            try:
                client.ping();break
            except redis.RedisError: time.sleep(.02)
        require_email_capacity(env.settings)
        with pytest.raises(ToolExecutionError) as error: require_email_capacity(env.settings.model_copy())
        assert error.value.code=='RATE_LIMITED'
        assert error.value.retry_after>0
        client.close()
    finally:
        process.terminate();process.wait(timeout=5)
        directory.cleanup()
    with pytest.raises(ToolExecutionError) as error: require_email_capacity(env.settings)
    assert error.value.code=='RATE_LIMIT_UNAVAILABLE'


def test_migration_from_legacy_columns_keeps_simulation_separate(env,tmp_path):
    # Old tables deliberately lack new columns/constraints; unrelated data is retained.
    legacy=env.engine if os.environ.get('TEST_TOOL_MYSQL_URL') else create_engine(f'sqlite:///{tmp_path}/legacy.sqlite')
    if legacy is env.engine:
        with legacy.begin() as conn:
            conn.execute(text('DROP TABLE tool_jobs'))
            conn.execute(text('DROP TABLE alert_records'))
    with legacy.begin() as conn:
        conn.execute(text('CREATE TABLE tool_jobs (id INTEGER PRIMARY KEY, report_id INTEGER, kind VARCHAR(64), status VARCHAR(32), attempts INTEGER, max_attempts INTEGER, depends_on_job_id INTEGER, run_after DATETIME, last_error TEXT, created_at DATETIME, updated_at DATETIME)'))
        conn.execute(text('CREATE TABLE alert_records (id INTEGER PRIMARY KEY, report_id INTEGER, channel VARCHAR(64), recipient VARCHAR(256), status VARCHAR(32), message TEXT, created_at DATETIME)'))
        conn.execute(text("INSERT INTO tool_jobs(id,report_id,kind,status) VALUES(1,7,'ALERT_SEND','SUCCESS'),(2,7,'ALERT_SEND','PENDING')"))
        conn.execute(text("INSERT INTO alert_records(id,report_id,channel,recipient,status,message) VALUES(1,7,'email','log','SUCCESS','deliveryMode=log')"))
    env.settings.alert_email_delivery_mode='smtp'
    migrate_tool_schema(legacy,env.settings)
    migrate_tool_schema(legacy,env.settings)
    with legacy.connect() as conn:
        rows=conn.execute(text('SELECT status,idempotency_key FROM tool_jobs ORDER BY id')).all()
        assert rows[0][0]=='SIMULATED'
        assert rows[1][0]=='DEAD'
        from app.services.tool_execution import operation_key
        assert rows[0][1]!=operation_key('ALERT_SEND',7,env.settings)
        assert conn.execute(text('SELECT status FROM alert_records')).scalar()=='SIMULATED'
    legacy.dispose()


def test_worker_process_crash_recovers_without_resetting_live_lease(env):
    import subprocess
    import sys
    import time
    env.settings.tool_queue_lease_seconds=.2
    with env.factory() as db:
        job_id=ToolQueueService(db,env.settings).enqueue_report(2,'LOW')[0].id
    child_env=os.environ.copy()
    child_env['DATABASE_URL']=env.settings.database_url
    child_env['TOOL_QUEUE_LEASE_SECONDS']='0.2'
    child_env['AUDIT_JOB_ID']=str(job_id)
    subprocess.run([sys.executable,'-c',
        'import os; from app.services.tool_queue import ToolQueueWorker; from app.core.config import get_settings; '
        'worker=ToolQueueWorker(get_settings()); assert worker._claim_job(int(os.environ["AUDIT_JOB_ID"])); os._exit(0)'],env=child_env,check=True)
    time.sleep(.25)
    worker=ToolQueueWorker(env.settings,env.factory)
    try:
        worker._recover_running_jobs()
        with env.factory() as db:
            assert db.get(ToolJob,job_id).status=='RETRY_WAIT'
        assert worker._claim_job(job_id)
    finally: worker.stop()


@pytest.mark.parametrize('smtp_code, retryable', [(421,True),(550,False)])
def test_smtp_rejection_retry_classification(env,monkeypatch,smtp_code,retryable):
    import smtplib
    env.settings.alert_email_delivery_mode='smtp'; env.settings.smtp_host='synthetic.invalid'
    env.settings.alert_email_to='synthetic@example.invalid'; env.settings.alert_email_from='test@example.invalid'
    def reject(*args): raise smtplib.SMTPDataError(smtp_code,b'synthetic rejection')
    monkeypatch.setattr(ToolOrchestrationService,'_send_alert_email',reject)
    with env.factory() as db:
        with pytest.raises(ToolExecutionError) as error: execute_tool(db,env.settings,'RISK_ALERT',1)
        assert error.value.retryable is retryable
        assert db.query(AlertRecord).one().status=='FAILED'


def test_admin_job_detail_and_replay_authorization(env):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes import router
    from app.core.database import get_db
    from app.core.security import current_user
    app=FastAPI();app.include_router(router)
    def session():
        with env.factory() as db: yield db
    app.dependency_overrides[get_db]=session
    with env.factory() as db:
        job=ToolQueueService(db,env.settings).enqueue_report(2,'LOW')[0]
        job.status='DEAD'; db.commit(); job_id=job.id
    with TestClient(app) as client:
        assert client.get(f'/api/admin/tool-jobs/{job_id}').status_code==401
        app.dependency_overrides[current_user]=lambda:SimpleNamespace(roles=['ROLE_USER'],username='student')
        assert client.post(f'/api/admin/tool-jobs/{job_id}/replay',json={'reason':'test'}).status_code==403
        app.dependency_overrides[current_user]=lambda:SimpleNamespace(roles=['ROLE_ADMIN'],username='admin')
        detail=client.get(f'/api/admin/tool-jobs/{job_id}')
        assert detail.status_code==200
        assert detail.json()['job']['idempotencyKey']
        response=client.post(f'/api/admin/tool-jobs/{job_id}/replay',json={'reason':'corrected configuration'})
        assert response.status_code==200
        assert response.json()['status']=='PENDING'


def test_dispatcher_keeps_case_capacity_independent_of_excel_backlog(env):
    import threading
    import time
    release=threading.Event();case_executed=threading.Event()
    class ControlledClient:
        def __init__(self,*args): pass
        async def execute_job(self,job,case_id):
            if job.kind=='EXCEL_REPORT':
                assert release.wait(5), 'test did not release Excel task'
            else:
                case_executed.set()
            return {'status':'SUCCESS'}
    env.settings.tool_queue_batch_size=1
    env.settings.tool_queue_poll_interval_seconds=.02
    with env.factory() as db:
        # Older Excel work must not occupy the scan budget for a newer case.
        ToolQueueService(db,env.settings).enqueue_report(2,'LOW')
        report=db.get(PsychologicalReport,1);report.risk_level='MEDIUM';db.commit()
        jobs=ToolQueueService(db,env.settings).enqueue_report(1,'MEDIUM')
        case_id=next(j.id for j in jobs if j.kind=='CASE_CREATE')
    worker=ToolQueueWorker(env.settings,env.factory,ControlledClient)
    try:
        worker.start()
        assert case_executed.wait(4), 'Excel backlog starved case creation'
    finally:
        release.set()
        worker.stop()
        for executor in worker.executors.values(): executor.shutdown(wait=True)
    with env.factory() as db:
        assert db.get(ToolJob,case_id).status=='SUCCESS'


def test_two_real_mcp_clients_dedupe_export(env):
    async def run():
        await asyncio.gather(*(MindBridgeMcpToolClient(env.settings).handle_report(2,'LOW') for _ in range(2)))
    asyncio.run(run())
    with env.factory() as db:
        assert db.query(ExcelRecord).count()==1
    workbook=load_workbook(env.settings.excel_path)
    assert sum(row[0]==2 for row in workbook.active.iter_rows(values_only=True))==1
    workbook.close()
