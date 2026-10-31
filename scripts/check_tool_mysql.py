"""Run selected integration tests against a disposable MySQL Docker container.

No existing database/volume is used; the container is removed in finally.
Requires a locally available mysql:8.0 image and Docker.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pymysql

root = Path(__file__).resolve().parents[1]
name = 'caretrace-tool-test-' + uuid.uuid4().hex[:10]
password = uuid.uuid4().hex
container = subprocess.check_output(['docker','run','--rm','-d','--name',name,
    '-e',f'MYSQL_ROOT_PASSWORD={password}','-e','MYSQL_DATABASE=caretrace_test_tools',
    '-p','127.0.0.1::3306','mysql:8.0'],text=True).strip()
try:
    ports=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .NetworkSettings.Ports}}',container],text=True))
    port=int(ports['3306/tcp'][0]['HostPort'])
    for _ in range(90):
        try:
            connection=pymysql.connect(host='127.0.0.1',port=port,user='root',password=password,database='caretrace_test_tools',connect_timeout=1)
            connection.close();break
        except pymysql.MySQLError: time.sleep(1)
    else: raise RuntimeError('Disposable MySQL did not become ready')
    env=os.environ.copy()
    env['TEST_TOOL_MYSQL_URL']=f'mysql+pymysql://root:{password}@127.0.0.1:{port}/caretrace_test_tools'
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_tool_pipeline.py'],cwd=root,env=env)
    raise SystemExit(result.returncode)
finally:
    subprocess.run(['docker','stop','-t','1',container],stdout=subprocess.DEVNULL,check=False)
