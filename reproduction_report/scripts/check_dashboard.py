"""Validate the offline dashboard in an isolated headless Edge instance via CDP."""
import base64
import json
import hashlib
import sys
from pathlib import Path
import time
import urllib.request

import websocket

ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
targets=json.load(urllib.request.urlopen('http://localhost:9335/json',timeout=10))
target=next(item for item in targets if item['type']=='page')
socket=websocket.create_connection(target['webSocketDebuggerUrl'],origin='http://localhost:9335',timeout=10)
sequence=0
errors=[]


def call(method,params=None):
    global sequence
    sequence+=1
    socket.send(json.dumps({'id':sequence,'method':method,'params':params or {}}))
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        item=json.loads(socket.recv())
        if item.get('method')=='Runtime.exceptionThrown':errors.append(item['params'])
        if item.get('method')=='Runtime.consoleAPICalled' and item['params']['type']=='error':errors.append(item['params'])
        if item.get('id')==sequence:
            if 'error' in item:raise RuntimeError(item['error'])
            return item.get('result',{})
    raise TimeoutError(method)


def evaluate(expression):
    value=call('Runtime.evaluate',{'expression':expression,'returnByValue':True})
    if value.get('exceptionDetails'):raise RuntimeError(value['exceptionDetails'])
    return value['result'].get('value')


call('Runtime.enable');call('Page.enable');call('Emulation.clearDeviceMetricsOverride')
call('Page.navigate',{'url':(ROOT/'index.html').as_uri()})
deadline=time.monotonic()+25
while time.monotonic()<deadline:
    ready=evaluate("['loss-chart','metric-chart','gpu-chart','progress-chart'].every(id=>document.getElementById(id)?._fullLayout)")
    if ready:break
    time.sleep(.5)
assert ready,'Plotly charts did not initialize'
checks={'title':evaluate('document.title'),'chart_count':4,
        'html_sha256':hashlib.sha256((ROOT/'index.html').read_bytes()).hexdigest(),
        'desktop_body_width':evaluate('document.body.scrollWidth'),
        'desktop_viewport_width':evaluate('document.documentElement.clientWidth'),
        'loss_trace_count':evaluate("document.getElementById('loss-chart')._fullData.length"),
        'gpu_trace_count':evaluate("document.getElementById('gpu-chart')._fullData.length"),
        'plotly_version':evaluate('Plotly.version'),'javascript_errors':errors}
assert checks['desktop_body_width']<=checks['desktop_viewport_width']+2
assert not errors
evaluate("document.getElementById('training').scrollIntoView(); document.querySelectorAll('img').forEach(img=>img.loading='eager'); true")
time.sleep(1)
capture=call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
(ROOT/'tools/dashboard_training_preview.png').write_bytes(base64.b64decode(capture['data']))
call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':False})
time.sleep(1)
checks['mobile_body_width']=evaluate('document.body.scrollWidth')
checks['mobile_viewport_width']=evaluate('document.documentElement.clientWidth')
assert checks['mobile_body_width']<=checks['mobile_viewport_width']+2
checks['status']='PASS'
(ROOT/'tools/dashboard_browser_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
socket.close()
