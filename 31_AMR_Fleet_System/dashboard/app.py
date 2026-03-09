"""app.py - Fleet 모니터링 웹 대시보드.

Flask 기반 단일 페이지 대시보드로 5대 AMR의 실시간 상태,
KPI 지표, 작업 대기열을 시각화한다.
ROS2 없이 독립 실행 가능 (Demo 모드).

실행:
    python dashboard/app.py          # http://localhost:8080
    python dashboard/app.py --port 9090
"""

import sys
import os
import json
import time
import math
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── 시뮬레이션 데이터 (Demo 모드) ─────────────────────────────

class FleetSimulator:
    """ROS2 없이 Fleet 상태를 시뮬레이션하는 데모 데이터 생성기."""

    WAREHOUSE_W, WAREHOUSE_H = 60.0, 40.0

    # 작업 위치 사전 정의
    LOCATIONS = {
        'Dock-1': (2, 35), 'Dock-2': (2, 30),
        'Dock-A': (58, 35), 'Dock-B': (58, 30),
        'Rack-A1': (8, 25), 'Rack-A3': (20, 25), 'Rack-A5': (32, 25),
        'Rack-B2': (14, 18), 'Rack-B4': (26, 18), 'Rack-B6': (38, 18),
        'Rack-C1': (8, 11), 'Rack-C3': (20, 11), 'Rack-C5': (32, 11),
        'Charging': (8, 3), 'Waiting': (50, 3),
    }

    def __init__(self):
        self.robots = {}
        self.tasks = []
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_counter = 0

        robot_starts = [
            (10, 20), (20, 15), (30, 25), (40, 10), (50, 20)]
        colors = ['#FF4444', '#4488FF', '#44CC44', '#FFCC00', '#CC44FF']

        for i in range(5):
            rid = f'amr_{i+1:02d}'
            self.robots[rid] = {
                'id': rid,
                'position': list(robot_starts[i]) + [0.0],
                'target': None,
                'battery': random.uniform(60, 100),
                'state': 'idle',
                'current_task': None,
                'color': colors[i],
                'trail': [list(robot_starts[i])],
                'speed': 0.0,
            }

        # 초기 작업 생성
        for _ in range(8):
            self._generate_task()

    def _generate_task(self):
        locs = list(self.LOCATIONS.keys())
        src = random.choice(locs[:4])   # Dock
        dst = random.choice(locs[4:13]) # Rack
        self.task_counter += 1
        self.tasks.append({
            'id': f'TASK-{self.task_counter:04d}',
            'from': src,
            'to': dst,
            'from_position': list(self.LOCATIONS[src]),
            'to_position': list(self.LOCATIONS[dst]),
            'priority': random.choice(['high', 'medium', 'low']),
            'status': 'pending',
            'assigned_robot': None,
            'created_at': time.time(),
        })

    def _reset_cycle(self):
        """50건 이상 완료 시 새 사이클 시작."""
        self.tasks = [t for t in self.tasks if t['status'] == 'in_progress']
        self.completed_tasks = 0
        self.start_time = time.time()

        robot_starts = [
            (10, 20), (20, 15), (30, 25), (40, 10), (50, 20)]
        for i, (rid, robot) in enumerate(self.robots.items()):
            if robot['state'] == 'idle':
                robot['position'] = list(robot_starts[i]) + [0.0]
                robot['trail'] = [list(robot_starts[i])]
                robot['battery'] = random.uniform(70, 100)

        for _ in range(8):
            self._generate_task()

    def update(self):
        """1 프레임 업데이트."""
        # 50건 이상 완료 시 새 사이클
        if self.completed_tasks >= 50:
            self._reset_cycle()

        # 대기 작업이 없으면 자동 생성
        pending = sum(1 for t in self.tasks if t['status'] == 'pending')
        if pending == 0:
            for _ in range(random.randint(2, 4)):
                self._generate_task()

        for rid, robot in self.robots.items():
            if robot['state'] == 'idle':
                # 대기 중인 작업 할당
                for task in self.tasks:
                    if task['status'] == 'pending':
                        task['status'] = 'in_progress'
                        task['assigned_robot'] = rid
                        robot['state'] = 'navigating'
                        robot['current_task'] = task['id']
                        robot['target'] = task['to_position']
                        break

            elif robot['state'] == 'navigating' and robot['target']:
                # 목표를 향해 이동
                dx = robot['target'][0] - robot['position'][0]
                dy = robot['target'][1] - robot['position'][1]
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < 0.5:
                    # 도착
                    robot['position'][0] = robot['target'][0]
                    robot['position'][1] = robot['target'][1]
                    robot['state'] = 'docking'
                    robot['speed'] = 0.0
                else:
                    speed = min(1.5, dist * 0.5)
                    robot['position'][0] += (dx / dist) * speed * 0.3
                    robot['position'][1] += (dy / dist) * speed * 0.3
                    robot['position'][2] = math.atan2(dy, dx)
                    robot['speed'] = speed
                    robot['battery'] = max(5, robot['battery'] - 0.01)

                robot['trail'].append(
                    [robot['position'][0], robot['position'][1]])
                if len(robot['trail']) > 100:
                    robot['trail'] = robot['trail'][-100:]

            elif robot['state'] == 'docking':
                # 도킹 완료 → 작업 완료
                for task in self.tasks:
                    if task['id'] == robot['current_task']:
                        task['status'] = 'completed'
                        break
                robot['state'] = 'idle'
                robot['current_task'] = None
                robot['target'] = None
                self.completed_tasks += 1

                # 새 작업 생성
                self._generate_task()

    def get_robots(self):
        return list(self.robots.values())

    def get_kpi(self):
        elapsed = max(1, time.time() - self.start_time)
        hours = elapsed / 3600.0
        active = sum(
            1 for r in self.robots.values() if r['state'] != 'idle')
        utilization = active / len(self.robots) * 100

        # 평균 작업 시간 계산
        completed = [t for t in self.tasks if t['status'] == 'completed']
        if completed:
            avg_time = elapsed / len(completed)
            avg_min = int(avg_time // 60)
            avg_sec = int(avg_time % 60)
            avg_task_time = f'{avg_min}분 {avg_sec}초'
        else:
            avg_task_time = '--'

        return {
            'tasks_per_hour': round(
                self.completed_tasks / max(0.01, hours), 1),
            'utilization': round(utilization, 1),
            'active_robots': active,
            'total_robots': len(self.robots),
            'deadlocks': 0,
            'completed_tasks': self.completed_tasks,
            'pending_tasks': sum(
                1 for t in self.tasks if t['status'] == 'pending'),
            'elapsed_minutes': round(elapsed / 60, 1),
            'avg_task_time': avg_task_time,
        }

    def get_alerts(self):
        """이상 상황 알림 목록."""
        alerts = []
        for rid, robot in self.robots.items():
            if robot['battery'] < 20:
                alerts.append({
                    'type': 'warning',
                    'message': f'{rid} 배터리 부족 ({robot["battery"]:.0f}%)',
                    'time': time.time(),
                })
            if robot['battery'] < 10:
                alerts.append({
                    'type': 'critical',
                    'message': f'{rid} 배터리 위험 - 충전 필요',
                    'time': time.time(),
                })
        return alerts[-10:]

    def get_tasks(self):
        # 최근 20개
        return sorted(
            self.tasks, key=lambda t: t['created_at'], reverse=True)[:20]

    def get_map_data(self):
        """물류센터 레이아웃 데이터."""
        shelves = []
        for name, pos in self.LOCATIONS.items():
            if name.startswith('Rack'):
                shelves.append({
                    'name': name, 'x': pos[0], 'y': pos[1],
                    'w': 4, 'h': 2})
        return {
            'width': self.WAREHOUSE_W,
            'height': self.WAREHOUSE_H,
            'shelves': shelves,
            'docks': [
                {'name': k, 'x': v[0], 'y': v[1]}
                for k, v in self.LOCATIONS.items()
                if k.startswith('Dock')],
            'charging': self.LOCATIONS['Charging'],
        }


# ─── HTTP 서버 ──────────────────────────────────────────────

simulator = FleetSimulator()


def _update_loop():
    """백그라운드에서 시뮬레이션 업데이트."""
    while True:
        simulator.update()
        time.sleep(0.3)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AMR Fleet 모니터링 대시보드</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#1a1a2e; color:#e0e0e0; font-family:'Segoe UI','Malgun Gothic',sans-serif; }
.header { background:#16213e; padding:10px 24px; display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #0f3460; }
.header h1 { font-size:18px; color:#4cc9f0; }
.header .status { font-size:12px; color:#8d99ae; }
.main { display:grid; grid-template-columns:1fr 1fr 1fr; grid-template-rows:auto 1fr 1fr; gap:10px; padding:10px; height:calc(100vh - 50px); }
.panel { background:#16213e; border-radius:8px; padding:12px; border:1px solid #0f3460; overflow:hidden; display:flex; flex-direction:column; }
.panel h2 { font-size:12px; color:#4cc9f0; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }
canvas { width:100%; flex:1; border-radius:4px; background:#0a0a1a; }

/* 로봇 상태 패널 - 상단 전체 */
.robot-status { grid-column: 1 / -1; }
.robot-cards { display:flex; gap:8px; flex:1; }
.robot-card { flex:1; background:#0f3460; border-radius:8px; padding:10px; display:flex; flex-direction:column; gap:4px; position:relative; }
.robot-card .name { font-size:13px; font-weight:bold; }
.robot-card .state { font-size:11px; padding:2px 6px; border-radius:8px; display:inline-block; width:fit-content; }
.robot-card .state.idle { background:#3d3d5c; color:#8d99ae; }
.robot-card .state.navigating { background:#1a5276; color:#5dade2; }
.robot-card .state.docking { background:#7b5e00; color:#ffd166; }
.robot-card .info { font-size:11px; color:#8d99ae; }
.robot-card .battery-bar { height:4px; background:#2d2d4e; border-radius:2px; margin-top:2px; }
.robot-card .battery-fill { height:100%; border-radius:2px; transition:width 0.3s; }

/* 창고 맵 */
.map-panel { grid-column: 1 / 3; grid-row: 2 / 4; }

/* KPI */
.kpi-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; flex:1; }
.kpi-card { background:#0f3460; border-radius:8px; padding:12px; text-align:center; display:flex; flex-direction:column; justify-content:center; }
.kpi-card .value { font-size:28px; font-weight:bold; color:#4cc9f0; }
.kpi-card .label { font-size:11px; color:#8d99ae; margin-top:2px; }
.kpi-card.alert .value { color:#ff6b6b; }

/* 알림 */
.alert-list { flex:1; overflow-y:auto; }
.alert-item { padding:6px 8px; margin-bottom:4px; border-radius:6px; font-size:11px; display:flex; align-items:center; gap:6px; }
.alert-item.warning { background:#7b5e00; color:#ffd166; }
.alert-item.critical { background:#6a040f; color:#ff758f; }
.alert-item.info { background:#1a5276; color:#5dade2; }
.alert-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.alert-item.warning .alert-dot { background:#ffd166; }
.alert-item.critical .alert-dot { background:#ff758f; }
.alert-item.info .alert-dot { background:#5dade2; }

/* 작업 테이블 */
table { width:100%; border-collapse:collapse; font-size:11px; }
th { background:#0f3460; padding:6px; text-align:left; color:#4cc9f0; position:sticky; top:0; }
td { padding:5px 6px; border-bottom:1px solid #1a1a3e; }
.badge { padding:2px 6px; border-radius:8px; font-size:10px; font-weight:bold; }
.badge.completed { background:#2d6a4f; color:#95d5b2; }
.badge.in_progress { background:#7b5e00; color:#ffd166; }
.badge.pending { background:#3d3d5c; color:#8d99ae; }
.badge.high { background:#6a040f; color:#ff758f; }
.badge.medium { background:#7b5e00; color:#ffd166; }
.badge.low { background:#2d4a3e; color:#95d5b2; }
.task-table-wrap { flex:1; overflow-y:auto; }
.chart-wrap { flex:1; position:relative; }
</style>
</head>
<body>
<div class="header">
  <h1>AMR Fleet 모니터링 대시보드</h1>
  <div class="status" id="clock">--</div>
</div>
<div class="main">
  <!-- 상단: 로봇 상태 카드 5개 -->
  <div class="panel robot-status">
    <h2>로봇 상태</h2>
    <div class="robot-cards" id="robotCards"></div>
  </div>

  <!-- 좌측: 창고 맵 (2x2 크기) -->
  <div class="panel map-panel">
    <h2>창고 맵 (60m x 40m)</h2>
    <canvas id="mapCanvas"></canvas>
  </div>

  <!-- 우측 상단: KPI -->
  <div class="panel">
    <h2>핵심 성과 지표 (KPI)</h2>
    <div class="kpi-grid">
      <div class="kpi-card"><div class="value" id="kpi-tph">--</div><div class="label">시간당 처리량</div></div>
      <div class="kpi-card"><div class="value" id="kpi-util">--</div><div class="label">로봇 가동률</div></div>
      <div class="kpi-card"><div class="value" id="kpi-avg">--</div><div class="label">평균 작업 시간</div></div>
      <div class="kpi-card" id="kpi-dl-card"><div class="value" id="kpi-dl">--</div><div class="label">교착 상태</div></div>
      <div class="kpi-card"><div class="value" id="kpi-done">--</div><div class="label">완료 작업</div></div>
      <div class="kpi-card"><div class="value" id="kpi-pending">--</div><div class="label">대기 작업</div></div>
    </div>
  </div>

  <!-- 우측 하단: 알림 + 작업 큐 탭 -->
  <div class="panel">
    <div style="display:flex; gap:12px; margin-bottom:8px;">
      <h2 id="tabAlert" style="cursor:pointer; opacity:1;" onclick="showTab('alert')">이상 알림</h2>
      <h2 id="tabTask" style="cursor:pointer; opacity:0.4;" onclick="showTab('task')">작업 큐</h2>
      <h2 id="tabChart" style="cursor:pointer; opacity:0.4;" onclick="showTab('chart')">처리 추이</h2>
    </div>
    <div id="alertPanel" class="alert-list"></div>
    <div id="taskPanel" class="task-table-wrap" style="display:none;">
      <table><thead><tr><th>상태</th><th>작업 ID</th><th>로봇</th><th>출발</th><th>도착</th><th>우선순위</th></tr></thead>
      <tbody id="taskBody"></tbody></table>
    </div>
    <div id="chartPanel" class="chart-wrap" style="display:none;"><canvas id="chartCanvas"></canvas></div>
  </div>
</div>

<script>
const mapCanvas = document.getElementById('mapCanvas');
const chartCanvas = document.getElementById('chartCanvas');
const mapCtx = mapCanvas.getContext('2d');
const chartCtx = chartCanvas.getContext('2d');
let chartData = [];
let currentTab = 'alert';

function showTab(tab) {
  currentTab = tab;
  document.getElementById('alertPanel').style.display = tab === 'alert' ? '' : 'none';
  document.getElementById('taskPanel').style.display = tab === 'task' ? '' : 'none';
  document.getElementById('chartPanel').style.display = tab === 'chart' ? '' : 'none';
  document.getElementById('tabAlert').style.opacity = tab === 'alert' ? 1 : 0.4;
  document.getElementById('tabTask').style.opacity = tab === 'task' ? 1 : 0.4;
  document.getElementById('tabChart').style.opacity = tab === 'chart' ? 1 : 0.4;
  if (tab === 'chart') resizeChart();
}

function resizeChart() {
  chartCanvas.width = chartCanvas.clientWidth * devicePixelRatio;
  chartCanvas.height = chartCanvas.clientHeight * devicePixelRatio;
  chartCtx.scale(devicePixelRatio, devicePixelRatio);
}

function resize() {
  mapCanvas.width = mapCanvas.clientWidth * devicePixelRatio;
  mapCanvas.height = mapCanvas.clientHeight * devicePixelRatio;
  mapCtx.scale(devicePixelRatio, devicePixelRatio);
  if (currentTab === 'chart') resizeChart();
}
window.addEventListener('resize', resize);
resize();

const STATE_KR = { idle: '대기', navigating: '주행 중', docking: '도킹 중' };

function updateRobotCards(robots) {
  const container = document.getElementById('robotCards');
  container.innerHTML = robots.map(r => {
    const bat = r.battery.toFixed(0);
    const batColor = bat > 50 ? '#44CC44' : bat > 20 ? '#FFCC00' : '#FF4444';
    const stateKr = STATE_KR[r.state] || r.state;
    const taskStr = r.current_task ? r.current_task : '--';
    const speedStr = r.speed ? r.speed.toFixed(1) + ' m/s' : '0.0 m/s';
    return `<div class="robot-card" style="border-left:3px solid ${r.color}">
      <div class="name" style="color:${r.color}">${r.id.toUpperCase()}</div>
      <div class="state ${r.state}">${stateKr}</div>
      <div class="info">작업: ${taskStr}</div>
      <div class="info">속도: ${speedStr}</div>
      <div class="info">배터리: ${bat}%</div>
      <div class="battery-bar"><div class="battery-fill" style="width:${bat}%; background:${batColor}"></div></div>
    </div>`;
  }).join('');
}

function updateAlerts(alerts, robots) {
  const panel = document.getElementById('alertPanel');
  let items = [];

  // 시스템 알림 생성
  const active = robots.filter(r => r.state !== 'idle').length;
  if (active === robots.length) {
    items.push({type:'info', msg:'모든 로봇 가동 중'});
  }
  robots.forEach(r => {
    if (r.battery < 20) items.push({type:'warning', msg:`${r.id.toUpperCase()} 배터리 부족 (${r.battery.toFixed(0)}%)`});
    if (r.battery < 10) items.push({type:'critical', msg:`${r.id.toUpperCase()} 배터리 위험 - 충전 필요`});
    if (r.state === 'docking') items.push({type:'info', msg:`${r.id.toUpperCase()} 도킹 수행 중`});
  });

  if (items.length === 0) items.push({type:'info', msg:'이상 없음 - 시스템 정상 가동'});

  const now = new Date().toLocaleTimeString('ko-KR', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  panel.innerHTML = items.map(a =>
    `<div class="alert-item ${a.type}"><div class="alert-dot"></div><span>${now}</span> ${a.msg}</div>`
  ).join('');
}

function drawMap(robots, mapData) {
  const ctx = mapCtx;
  const W = mapCanvas.clientWidth, H = mapCanvas.clientHeight;
  const sx = W / mapData.width, sy = H / mapData.height;
  ctx.clearRect(0, 0, W, H);

  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  // 그리드
  ctx.strokeStyle = '#252545'; ctx.lineWidth = 0.5;
  for (let x = 0; x < mapData.width; x += 5) { ctx.beginPath(); ctx.moveTo(x*sx,0); ctx.lineTo(x*sx,H); ctx.stroke(); }
  for (let y = 0; y < mapData.height; y += 5) { ctx.beginPath(); ctx.moveTo(0,y*sy); ctx.lineTo(W,y*sy); ctx.stroke(); }

  // 선반
  ctx.fillStyle = '#2d3a5c'; ctx.strokeStyle = '#4a5a8c'; ctx.lineWidth = 1;
  (mapData.shelves || []).forEach(s => {
    const rx = s.x*sx, ry = (mapData.height-s.y)*sy;
    ctx.fillRect(rx-s.w*sx/2, ry-s.h*sy/2, s.w*sx, s.h*sy);
    ctx.strokeRect(rx-s.w*sx/2, ry-s.h*sy/2, s.w*sx, s.h*sy);
    ctx.fillStyle='#5a6a9c'; ctx.font='9px sans-serif'; ctx.textAlign='center';
    ctx.fillText(s.name, rx, ry+3); ctx.fillStyle='#2d3a5c';
  });

  // 도크
  (mapData.docks||[]).forEach(d => {
    const dx=d.x*sx, dy=(mapData.height-d.y)*sy;
    ctx.fillStyle='#0f3460'; ctx.fillRect(dx-12,dy-8,24,16);
    ctx.fillStyle='#4cc9f0'; ctx.font='9px sans-serif'; ctx.textAlign='center';
    ctx.fillText(d.name, dx, dy+3);
  });

  // 충전소
  if (mapData.charging) {
    const cx=mapData.charging[0]*sx, cy=(mapData.height-mapData.charging[1])*sy;
    ctx.fillStyle='#44CC44'; ctx.font='10px sans-serif'; ctx.textAlign='center';
    ctx.fillText('⚡충전소', cx, cy+3);
  }

  // 로봇 궤적 + 위치
  robots.forEach(r => {
    const trail = r.trail || [];
    if (trail.length > 1) {
      ctx.strokeStyle = r.color+'44'; ctx.lineWidth = 2; ctx.beginPath();
      trail.forEach((p,i) => { const px=p[0]*sx, py=(mapData.height-p[1])*sy; i===0?ctx.moveTo(px,py):ctx.lineTo(px,py); });
      ctx.stroke();
    }

    // 목표 표시
    if (r.target) {
      const tx=r.target[0]*sx, ty=(mapData.height-r.target[1])*sy;
      ctx.strokeStyle=r.color+'88'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(r.position[0]*sx,(mapData.height-r.position[1])*sy); ctx.lineTo(tx,ty); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle=r.color; ctx.lineWidth=2;
      ctx.beginPath(); ctx.arc(tx,ty,5,0,Math.PI*2); ctx.stroke();
    }

    const rx=r.position[0]*sx, ry=(mapData.height-r.position[1])*sy;
    ctx.fillStyle=r.color; ctx.beginPath(); ctx.arc(rx,ry,7,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#fff'; ctx.font='bold 8px sans-serif'; ctx.textAlign='center';
    ctx.fillText(r.id.slice(-2), rx, ry+3);
    const angle=r.position[2]||0;
    ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(rx,ry);
    ctx.lineTo(rx+Math.cos(-angle)*12, ry+Math.sin(-angle)*12); ctx.stroke();
  });
}

function drawChart() {
  const ctx = chartCtx;
  const W = chartCanvas.clientWidth, H = chartCanvas.clientHeight;
  ctx.clearRect(0,0,W,H);
  if (chartData.length < 2) return;

  const maxVal = Math.max(...chartData.map(d=>d.v), 1);
  const padL=40, padB=24, padT=10, padR=10;
  const cw=W-padL-padR, ch=H-padT-padB;

  ctx.strokeStyle='#3d3d5c'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,H-padB); ctx.lineTo(W-padR,H-padB); ctx.stroke();

  ctx.fillStyle='#8d99ae'; ctx.font='10px sans-serif'; ctx.textAlign='right';
  for (let i=0;i<=4;i++) {
    const val=Math.round(maxVal*i/4), y=H-padB-(ch*i/4);
    ctx.fillText(val,padL-4,y+3);
    ctx.strokeStyle='#252545'; ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
  }

  ctx.strokeStyle='#4cc9f0'; ctx.lineWidth=2; ctx.beginPath();
  chartData.forEach((d,i) => { const x=padL+(i/(chartData.length-1))*cw, y=H-padB-(d.v/maxVal)*ch; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.stroke();

  const grad=ctx.createLinearGradient(0,padT,0,H-padB);
  grad.addColorStop(0,'rgba(76,201,240,0.3)'); grad.addColorStop(1,'rgba(76,201,240,0)');
  ctx.fillStyle=grad; ctx.beginPath();
  chartData.forEach((d,i) => { const x=padL+(i/(chartData.length-1))*cw, y=H-padB-(d.v/maxVal)*ch; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.lineTo(padL+cw,H-padB); ctx.lineTo(padL,H-padB); ctx.closePath(); ctx.fill();
}

const STATUS_KR = { completed:'완료', in_progress:'진행 중', pending:'대기' };
const PRIO_KR = { high:'긴급', medium:'보통', low:'낮음' };

function updateTasks(tasks) {
  const tbody = document.getElementById('taskBody');
  tbody.innerHTML = tasks.map(t => `<tr>
    <td><span class="badge ${t.status}">${STATUS_KR[t.status]||t.status}</span></td>
    <td>${t.id}</td>
    <td>${t.assigned_robot ? t.assigned_robot.toUpperCase() : '-'}</td>
    <td>${t.from}</td>
    <td>${t.to}</td>
    <td><span class="badge ${t.priority}">${PRIO_KR[t.priority]||t.priority}</span></td>
  </tr>`).join('');
}

async function refresh() {
  try {
    const [rRes,kRes,tRes,mRes,aRes] = await Promise.all([
      fetch('/api/robots'),fetch('/api/kpi'),fetch('/api/tasks'),fetch('/api/map'),fetch('/api/alerts')
    ]);
    const robots=await rRes.json(), kpi=await kRes.json(), tasks=await tRes.json(), mapData=await mRes.json(), alerts=await aRes.json();

    updateRobotCards(robots);
    drawMap(robots, mapData);
    updateTasks(tasks);
    updateAlerts(alerts, robots);

    document.getElementById('kpi-tph').textContent = kpi.tasks_per_hour;
    document.getElementById('kpi-util').textContent = kpi.utilization + '%';
    document.getElementById('kpi-avg').textContent = kpi.avg_task_time;
    document.getElementById('kpi-dl').textContent = kpi.deadlocks;
    document.getElementById('kpi-dl-card').className = 'kpi-card' + (kpi.deadlocks > 0 ? ' alert' : '');
    document.getElementById('kpi-done').textContent = kpi.completed_tasks;
    document.getElementById('kpi-pending').textContent = kpi.pending_tasks;

    chartData.push({v: kpi.completed_tasks});
    if (chartData.length > 60) chartData.shift();
    if (currentTab === 'chart') drawChart();

    document.getElementById('clock').textContent =
      new Date().toLocaleTimeString('ko-KR') + ' | 경과 ' +
      kpi.elapsed_minutes + '분 | 완료 ' + kpi.completed_tasks + '건';
  } catch(e) { console.error(e); }
}

setInterval(refresh, 750);
refresh();
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP 요청 핸들러."""

    def log_message(self, format, *args):
        pass  # 로그 억제

    def _respond_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(
            data, ensure_ascii=False).encode('utf-8'))

    def _respond_html(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self._respond_html(DASHBOARD_HTML)
        elif path == '/api/robots':
            self._respond_json(simulator.get_robots())
        elif path == '/api/kpi':
            self._respond_json(simulator.get_kpi())
        elif path == '/api/tasks':
            self._respond_json(simulator.get_tasks())
        elif path == '/api/map':
            self._respond_json(simulator.get_map_data())
        elif path == '/api/alerts':
            self._respond_json(simulator.get_alerts())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = 8888
    if '--port' in sys.argv:
        idx = sys.argv.index('--port')
        port = int(sys.argv[idx + 1])

    # 시뮬레이션 백그라운드 스레드
    t = threading.Thread(target=_update_loop, daemon=True)
    t.start()

    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f'AMR Fleet Dashboard: http://localhost:{port}')
    print('Ctrl+C로 종료')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n종료')
        server.server_close()


if __name__ == '__main__':
    main()
