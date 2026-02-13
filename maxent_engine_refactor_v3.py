import os
import threading
import subprocess
import time
import configparser
from datetime import datetime
from multiprocessing import Pool, freeze_support
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import hashlib
from multiprocessing import Manager

# -*- coding: utf-8 -*-

"""
[폴더 규약 요약]

# MaxEnt Auto Engine v1.4.9-R3 [Build 260213]
# Copyright (c) 2026 [Chi-young Choi]
# Licensed under the MIT License (see text below)
MIT License

Copyright (c) 2026 Chi-young CHOI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


System Directory and Data Workflow Specification

1. Input Specifications (Environmental Layers; ASCII Format)
The engine requires environmental variables in ESRI ASCII grid format, structured as follows:
Phase 1 (Baseline Model Training):
Standard Path: {base_dir}/02.ASCII_DATA/GCM-Current/SSP-Current/Present
Note: If cfg['phase1_custom_path'] is defined, the system prioritizes the user-specified directory.
Phase 2 (Future/Scenario Projection):
Path Structure: {base_dir}/02.ASCII_DATA/GCM-Future/{GCM}/{SSP}/{Period}

2. Output Specifications (Results and Metadata)
Output files are systematically organized to maintain data integrity and facilitate post-processing:

Phase 1: Model Training Artifacts
Path: {base_dir}/03.Results/{species}/GCM-Current/SSP-Current/Present
Core Outputs:
{species}.lambdas (or {species}_{i}.lambdas for multi-replicate runs).
model_meta.json: Contains critical metadata including option fingerprints, replicate counts, and the unique model_id (SHA-256).

Phase 2: Spatial Projection Results (ASCII)
Path: {base_dir}/03.Results/{species}/GCM-Future/{GCM}/{SSP}/{Period}
Core Outputs:
{species}.asc (or {species}_{i}.asc for multi-replicate runs).

Phase 3: Ensemble Statistical Analysis
Location: Generated within the corresponding Phase 2 output directory.
Statistical Metrics:
{species}_{period}_avg.asc: Arithmetic Mean
{species}_{period}_max.asc: Maximum Value
{species}_{period}_min.asc: Minimum Value
{species}_{period}_median.asc: Median Value
{species}_{period}_stddev.asc: Standard Deviation

입력(환경 레이어; ASCII):
  - Phase 1(Current 학습): {base_dir}/02.ASCII_DATA/GCM-Current/SSP-Current/Present
    * cfg['phase1_custom_path']가 존재하면 해당 경로를 우선 사용
  - Phase 2(미래/시나리오 투영): {base_dir}/02.ASCII_DATA/GCM-Future/{GCM}/{SSP}/{Period}

출력:
  - Phase 1 결과(학습 산출물): {base_dir}/03.Results/{species}/GCM-Current/SSP-Current/Present
    * {species}.lambdas 또는 {species}_{i}.lambdas (replicates>1)
    * model_meta.json (옵션 fingerprint/replicates/model_id 기록)
  - Phase 2 결과(투영 asc): {base_dir}/03.Results/{species}/GCM-Future/{GCM}/{SSP}/{Period}
    * {species}.asc 또는 {species}_{i}.asc (replicates>1)
1)
  - Phase 3 통계(asc 통합): Phase 2 출력 폴더 내에 생성
    * {species}_{period}_avg.asc / _max.asc / _min.asc / _median.asc / _stddev.asc

"""


# ------------------------------------------------------------
# 1) 실행 엔진 (Parallel Backend)
# ------------------------------------------------------------

def run_phase1_train(args):
    """Phase 1: 현재 기후 데이터를 이용해 기준 모델 생성 (0~N 번호 인식)"""
    try:
        job, cfg, opt = args

        custom_path = cfg.get('phase1_custom_path', "").strip()
        if custom_path and os.path.exists(custom_path):
            env_path = os.path.normpath(custom_path)
        else:
            env_path = os.path.normpath(
                os.path.join(cfg['base_dir'], "02.ASCII_DATA", "GCM-Current", "SSP-Current", "Present")
            )

        out_dir = os.path.normpath(
            os.path.join(cfg['base_dir'], "03.Results", job['species'], "GCM-Current", "SSP-Current", "Present")
        )


        if not os.path.exists(env_path):
            dbg = job.get('env_path_debug', env_path)
            return {"status": "FAILED", "job": job, "error": f"환경 레이어 없음: {env_path} (debug:{dbg})"}

        os.makedirs(out_dir, exist_ok=True)
        target_reps = int(opt['reps'])

        # Phase1 옵션/학습 설정 fingerprint (학습 결과 재사용/경고 판단용)
        impact_keys = [
            'linear', 'quadratic', 'hinge', 'product', 'threshold', 'autofeature',
            'beta', 'convergence', 'reps', 'rep_type', 'test_pts', 'max_iter',
            'clamping', 'extrapolate',
        ]
        current_fp = {k: opt.get(k) for k in impact_keys if k in opt}

        def check_files_exist(directory, species, count):
            if count > 1:
                for i in range(count):
                    f_path = os.path.join(directory, f"{species}_{i}.lambdas")
                    if not (os.path.exists(f_path) and os.path.getsize(f_path) > 0):
                        return False
                return True
            else:
                f_path = os.path.join(directory, f"{species}.lambdas")
                return os.path.exists(f_path) and os.path.getsize(f_path) > 0

        def _compute_model_id(train_dir, species, reps):
            h = hashlib.sha256()
            if reps > 1:
                lambda_files = [os.path.join(train_dir, f"{species}_{i}.lambdas") for i in range(reps)]
            else:
                lambda_files = [os.path.join(train_dir, f"{species}.lambdas")]

            for fp in lambda_files:
                if not os.path.exists(fp):
                    return None
                with open(fp, 'rb') as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
            return h.hexdigest()

        origin_meta_path = os.path.join(out_dir, "model_meta.json")

        def _load_existing_meta():
            if not os.path.exists(origin_meta_path):
                return None
            try:
                with open(origin_meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None

        def _write_meta():
            model_id = _compute_model_id(out_dir, job['species'], target_reps)
            train_meta = {
                "species": job['species'],
                "fingerprint": current_fp,
                "target_reps": target_reps,
                "model_id": model_id,
                "train_time": time.time(),
            }
            with open(origin_meta_path, 'w', encoding='utf-8') as f:
                json.dump(train_meta, f, indent=4)

        if check_files_exist(out_dir, job['species'], target_reps):
            existing_meta = _load_existing_meta()
            if existing_meta:
                same_fp = (existing_meta.get('fingerprint') == current_fp)
                same_reps = (existing_meta.get('target_reps') == target_reps)
                # 모델 파일이 같더라도, 옵션이 바뀌었으면 사용자 확인이 필요
                if not (same_fp and same_reps):
                    return {
                        "status": "NEED_CONFIRM",
                        "job": job,
                        "existing_meta": existing_meta,
                        "current_fingerprint": current_fp,
                    }
            else:
                # 메타가 없으면 일단 갱신만 하고 넘어감
                _write_meta()
            return {"status": "SKIPPED", "job": job}

        cmd = [
            "java", f"-{cfg['memory']}", "-jar", cfg['jar_path'],
            f"samplesfile={os.path.join(cfg['base_dir'], '01.SpeicesLIST', job['species'] + '.csv')}",
            f"environmentallayers={env_path}",
            f"outputdirectory={out_dir}",
            "autorun", "nowarnings", "notooltips",
            f"threads={cfg['java_threads']}",
            f"linear={opt['linear']}",
            f"quadratic={opt['quadratic']}",
            f"hinge={opt['hinge']}",
            f"product={opt['product']}",
            f"threshold={opt['threshold']}",
            f"autofeature={opt['autofeature']}",
            f"betamultiplier={opt['beta']}",
            f"convergencethreshold={opt['convergence']}",
            f"replicates={opt['reps']}",
            f"replicatetype={opt['rep_type']}",
            f"randomtestpoints={opt['test_pts']}",
            f"maximumiterations={opt['max_iter']}",
            f"maximumbackground={opt['max_bg']}",
            f"randomseed={opt['randomseed']}",
            f"outputformat={opt['out_format']}",
            "outputgrids=true",
            "askoverwrite=false",
            f"jackknife={opt['jackknife']}",
            f"responsecurves={opt['response']}",
            f"pictures={opt['plots']}",
            f"defaultprevalence={opt['prevalence']}",
            f"doclamp={opt['clamping']}",
            f"extrapolate={opt['extrapolate']}",
            f"writeplotdata={opt['writeplot']}",
            f"removeduplicates={opt['removeduplicates']}",
        ]

        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=si, text=True)
        time.sleep(0.5)

        if check_files_exist(out_dir, job['species'], target_reps):
            _write_meta()
            return {"status": "SUCCESS", "job": job}
        else:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            if result.returncode not in (0, None):
                err_msg = f"java 종료코드={result.returncode}\nSTDERR: {stderr[:800]}\nSTDOUT: {stdout[:800]}".strip()
            else:
                err_msg = stderr if stderr else "결과 람다 파일 생성 실패"
            return {"status": "FAILED", "job": job, "error": err_msg}

    except Exception as e:
        import traceback
        return {"status": "ERROR", "job": job, "error": f"{str(e)}\n{traceback.format_exc()}"}


def run_phase2_projection(args):
    """Phase 2: 이미 파일이 있다면 즉시 Skip하고, 실행 시에만 딜레이 적용"""
    try:
        job, cfg, opt = args
        # 동일 env_path(=같은 시나리오 폴더)에 대해 동시에 density.Project를 실행하면
        # MaxEnt cache(maxent.cache*.mxe) 경합으로 EOF/NPE가 발생할 수 있어 폴더 단위 락을 사용
        lock_map = job.get('_lock_map')
        target_file = os.path.normpath(job['target_file'])

        if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
            return {"status": "SKIPPED", "job": job}

        time.sleep(0.5 + (job.get('rep_index', 0) * 0.1))

        lambda_path = os.path.normpath(job['lambda_path'])
        env_path = os.path.normpath(job['proj_layers'])
        out_dir = os.path.normpath(job['out_dir'])
        # 폴더 단위 직렬화(락): 같은 env_path는 1개 프로세스만 실행
        env_lock = None
        if lock_map is not None:
            try:
                env_lock = lock_map.get(env_path)
            except Exception:
                env_lock = None

        if not os.path.exists(env_path):
            return {"status": "FAILED", "job": job, "error": f"환경 레이어 없음: {env_path}"}
        if not os.path.exists(lambda_path):
            return {"status": "FAILED", "job": job, "error": f"람다 파일 없음: {lambda_path}"}

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            time.sleep(0.1)

        cmd = ["java", f"-{cfg['memory']}", "-cp", cfg['jar_path'], "density.Project",
               lambda_path, env_path, target_file]

        if opt.get('clamping') == 'true':
            cmd.append("doclamp")
        if opt.get('extrapolate') == 'true':
            cmd.append("extrapolate")

        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=si, text=True)
        if env_lock is not None:
            try:
                env_lock.acquire()
            except Exception:
                env_lock = None

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=si, text=True)
        finally:
            if env_lock is not None:
                try:
                    env_lock.release()
                except Exception:
                    pass
                    
        success = False
        for _ in range(5):
            if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                success = True
                break
            time.sleep(0.5)

        if success:
            return {"status": "SUCCESS", "job": job}
        else:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            if proc.returncode not in (0, None):
                err_msg = f"java 종료코드={proc.returncode}\nSTDERR: {stderr[:800]}\nSTDOUT: {stdout[:800]}".strip()
            else:
                err_msg = stderr if stderr else "결과 파일 생성 실패"
            return {"status": "FAILED", "job": job, "error": err_msg}

    except Exception as e:
        import traceback
        return {"status": "ERROR", "job": job, "error": f"{str(e)}\n{traceback.format_exc()}"}


def run_phase3_averaging(args):
    """Phase 3: asc 통합 및 통계 세트(avg, max, min, median, stddev) 생성

    - 소규모 격자: 기존 방식(전체 적재) 사용
    - 대규모 격자/replicates: 타일(행 블록) 스트리밍으로 메모리 피크 방지
    """
    try:
        job, cfg, opt = args
        out_dir = os.path.join(cfg['base_dir'], "03.Results", job['species'], "GCM-Future", job['gcm'], job['ssp'], job['period'])
        target_reps = int(job.get('target_reps', 15))


        prefix = f"{job['species']}_{job['period']}"
        final_avg_file = os.path.join(out_dir, f"{prefix}_avg.asc")

        if os.path.exists(final_avg_file) and os.path.getsize(final_avg_file) > 0:
            return {"status": "SKIPPED", "job": job, "mode": "SKIP"}

        time.sleep(0.5)

        # 입력 파일 목록
        fpaths = []
        for i in range(target_reps):
            fpath = os.path.join(out_dir, f"{job['species']}_{i}.asc")
            if not os.path.exists(fpath):
                return {"status": "FAILED", "job": job, "error": f"누락된 파일: {job['species']}_{i}.asc"}
            fpaths.append(fpath)

        # 헤더/격자 크기 파싱
        with open(fpaths[0], 'r', encoding='utf-8', errors='ignore') as f0:
            header_lines = [next(f0) for _ in range(6)]
        header = "".join(header_lines)

        def _parse_dim(lines6):
            ncols = None
            nrows = None
            for ln in lines6:
                parts = ln.strip().split()
                if len(parts) >= 2:
                    k = parts[0].lower()
                    if k == 'ncols':
                        ncols = int(float(parts[1]))
                    elif k == 'nrows':
                        nrows = int(float(parts[1]))
            return ncols, nrows

        ncols, nrows = _parse_dim(header_lines)
        if not ncols or not nrows:
            # 헤더 파싱 실패 시 보수적으로 스트리밍 경로(행 단위)로 진행
            ncols = None
            nrows = None

        # 메모리 예측(대략): stack(float64) 1개 기준
        # 실제로는 mean/std 등 추가 배열이 더 생기므로 보수적으로 2배로 잡음
        use_streaming = False
        if ncols and nrows:
            est_bytes = int(target_reps * nrows * ncols * 8) * 2
            # 1.2GB 이상이면 스트리밍
            use_streaming = est_bytes >= int(1.2 * 1024**3)
        else:
            use_streaming = True

        if not use_streaming:
            # -------------------------
            # 기존 방식(전체 적재)
            # -------------------------
            all_data = []
            for fpath in fpaths:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                data = np.loadtxt(lines[6:], dtype=np.float32)
                all_data.append(data)

            stack = np.stack(all_data, axis=0)
            stats = {
                'avg': np.mean(stack, axis=0),
                'max': np.max(stack, axis=0),
                'min': np.min(stack, axis=0),
                'median': np.median(stack, axis=0),
                'stddev': np.std(stack, axis=0),
            }

            for name, data in stats.items():
                out_path = os.path.join(out_dir, f"{prefix}_{name}.asc")
                with open(out_path, 'w') as f:
                    f.write(header)
                    np.savetxt(f, data, fmt='%.6f')

            return {"status": "SUCCESS", "job": job, "mode": "FULL", "ncols": ncols, "nrows": nrows, "target_reps": target_reps}

        # -------------------------
        # 스트리밍(타일) 방식
        # -------------------------
        def _row_iter(fp):
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(6):
                    next(f, None)
                for line in f:
                    yield line

        # 타일 크기(행 단위) 결정: (reps * tile_rows * ncols * 4bytes) ~= 256MB 목표
        if ncols:
            target_tile_bytes = 256 * 1024 * 1024
            tile_rows = max(1, int(target_tile_bytes / max(1, target_reps * ncols * 4)))
            tile_rows = min(tile_rows, 2000)
        else:
            tile_rows = 256

        out_files = {}
        try:
            for name in ('avg', 'max', 'min', 'median', 'stddev'):
                out_path = os.path.join(out_dir, f"{prefix}_{name}.asc")
                f = open(out_path, 'w', encoding='utf-8')
                f.write(header)
                out_files[name] = f

            iters = [_row_iter(fp) for fp in fpaths]

            while True:
                # tile_rows 만큼 각 replicate에서 라인을 모음
                tile_lines = []
                got_any = False
                for _ in range(tile_rows):
                    rows = []
                    for it in iters:
                        try:
                            ln = next(it)
                        except StopIteration:
                            ln = None
                        rows.append(ln)
                    if rows[0] is None:
                        break
                    if any(r is None for r in rows):
                        raise RuntimeError("asc 파일 행 수 불일치(중간에 파일이 끝남)")
                    tile_lines.append(rows)
                    got_any = True

                if not got_any:
                    break

                # (reps, tile_rows, ncols) 블록 구성
                block_rows = []
                for rows in tile_lines:
                    arrs = [np.fromstring(r, sep=' ', dtype=np.float32) for r in rows]
                    block_rows.append(arrs)

                block = np.stack([np.stack(r, axis=0) for r in block_rows], axis=1)  # (reps, tile_rows, ncols)

                avg = np.mean(block, axis=0)
                vmax = np.max(block, axis=0)
                vmin = np.min(block, axis=0)
                med = np.median(block, axis=0)
                std = np.std(block, axis=0)

                np.savetxt(out_files['avg'], avg, fmt='%.6f')
                np.savetxt(out_files['max'], vmax, fmt='%.6f')
                np.savetxt(out_files['min'], vmin, fmt='%.6f')
                np.savetxt(out_files['median'], med, fmt='%.6f')
                np.savetxt(out_files['stddev'], std, fmt='%.6f')

            return {"status": "SUCCESS", "job": job, "mode": "STREAM", "tile_rows": tile_rows, "ncols": ncols, "nrows": nrows, "target_reps": target_reps}

        finally:
            for f in out_files.values():
                try:
                    f.close()
                except:
                    pass

    except Exception as e:
        import traceback
        return {"status": "ERROR", "job": job, "error": f"{str(e)}\n{traceback.format_exc()}"}


# ------------------------------------------------------------
# 2) GUI 클래스 (process()를 단계별 메서드로 분리한 버전)
# ------------------------------------------------------------

class MaxEntFullGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maxent_Auto_Engine_v1.4.9-R3_Build260213")

        self.root.geometry("920x1010")
        self.root.configure(bg="#f8f9fa")

        self.config_file = "maxent_Engine_config.ini"
        self.is_running = False
        self.pause_requested = False
        self.pool = None
        self._mgr = None

        self.setup_variables()
        self.setup_ui()
        self.load_config()
        self.refresh_all_data()

    # -------------------------
    # UI/설정 관련 (원본과 동일)
    # -------------------------

    def setup_variables(self):
        total_cores = os.cpu_count() or 4
        default_workers = max(1, int(total_cores * 0.4))

        self.phase1_custom_path = tk.StringVar(value="")
        self.jar_path = tk.StringVar(value="D:/Maxent/maxent.jar")
        self.base_dir = tk.StringVar(value="D:/Maxent")

        initial_p1_path = os.path.join(self.base_dir.get(), "02.ASCII_DATA", "GCM-Current", "SSP-Current", "Present")
        self.phase1_custom_path = tk.StringVar(value=os.path.normpath(initial_p1_path))

        self.base_dir.trace_add("write", self.update_phase1_path)

        self.workers = tk.IntVar(value=default_workers)
        self.java_threads = tk.IntVar(value=1)
        self.memory = tk.StringVar(value="mx2g")
        self.include_current = tk.BooleanVar(value=True)
        self.run_projection = tk.BooleanVar(value=True)

        # 실행 상태(버튼 토글용)
        self.run_state = tk.StringVar(value="IDLE")

        self.opt = {
            'linear': tk.BooleanVar(value=True), 'quadratic': tk.BooleanVar(value=True),
            'hinge': tk.BooleanVar(value=True), 'product': tk.BooleanVar(value=True),
            'threshold': tk.BooleanVar(value=True), 'autofeature': tk.BooleanVar(value=True),
            'beta': tk.DoubleVar(value=1.0), 'convergence': tk.StringVar(value="0.00001"),
            'reps': tk.IntVar(value=15), 'rep_type': tk.StringVar(value="subsample"),
            'test_pts': tk.IntVar(value=25), 'max_iter': tk.IntVar(value=5000),
            'max_bg': tk.IntVar(value=10000), 'prevalence': tk.DoubleVar(value=0.5),
            'out_format': tk.StringVar(value="logistic"), 'jackknife': tk.BooleanVar(value=False),
            'response': tk.BooleanVar(value=True), 'plots': tk.BooleanVar(value=True),
            'askoverwrite': tk.BooleanVar(value=False), 'clamping': tk.BooleanVar(value=True),
            'extrapolate': tk.BooleanVar(value=False), 'writeplot': tk.BooleanVar(value=True),
            'removeduplicates': tk.BooleanVar(value=True), 'randomseed': tk.BooleanVar(value=True),
            'outputgrids': tk.BooleanVar(value=True),
        }

        self.scen_vars = {"GCMs": {}, "SSPs": {}, "Periods": {}}
        self.sp_vars = {}

    def update_phase1_path(self, *args):
        new_base = self.base_dir.get()        
        if new_base:
            new_p1_path = os.path.normpath(os.path.join(new_base, "02.ASCII_DATA", "GCM-Current", "SSP-Current", "Present"))
            self.phase1_custom_path.set(new_p1_path)

    # -------------------------
    # UI / 설정 / 유틸
    # -------------------------

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.tab_main = tk.Frame(self.notebook, bg="white", padx=20, pady=15)
        self.tab_opt = tk.Frame(self.notebook, bg="white", padx=20, pady=15)
        self.notebook.add(self.tab_main, text="🧬 시나리오 및 종 선택")
        self.notebook.add(self.tab_opt, text="⚙️ MaxEnt 세부 옵션")
        self.notebook.pack(expand=True, fill="both", padx=15, pady=5)

        # [1] 시스템 설정
        p_frame = ttk.LabelFrame(self.tab_main, text=" 기본 시스템 설정 ", padding=10)
        p_frame.pack(fill="x", pady=5)

        tk.Label(p_frame, text="Base Dir:", bg="white").grid(row=0, column=0, sticky="w")
        tk.Entry(p_frame, textvariable=self.base_dir).grid(row=0, column=1, sticky="ew", padx=10)
        tk.Button(p_frame, text="경로선택", command=self.browse_base).grid(row=0, column=2)

        tk.Label(p_frame, text="JAR Path:", bg="white").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(p_frame, textvariable=self.jar_path).grid(row=1, column=1, sticky="ew", padx=10)
        tk.Button(p_frame, text="찾기", command=lambda: self.browse(self.jar_path)).grid(row=1, column=2)

        hw_frame = tk.Frame(p_frame, bg="white")
        hw_frame.grid(row=2, column=1, sticky="w")
        tk.Label(hw_frame, text="Workers:", bg="white").pack(side="left")
        tk.Spinbox(hw_frame, from_=1, to=64, textvariable=self.workers, width=6).pack(side="left", padx=5)
        tk.Label(hw_frame, text="Java Threads:", bg="white").pack(side="left", padx=(10, 0))
        tk.Spinbox(hw_frame, from_=1, to=32, textvariable=self.java_threads, width=6).pack(side="left", padx=5)
        tk.Label(hw_frame, text="RAM:", bg="white").pack(side="left", padx=(10, 0))
        tk.Entry(hw_frame, textvariable=self.memory, width=10).pack(side="left", padx=5)
        p_frame.columnconfigure(1, weight=1)

        # [2] 종 선택
        self.sp_outer_frame = ttk.LabelFrame(self.tab_main, text=" 분석 대상 종 선택 (01.SpeicesLIST) ", padding=10)
        self.sp_outer_frame.pack(fill="x", pady=5)

        self.sp_canvas = tk.Canvas(self.sp_outer_frame, height=110, bg="#fcfcfc", highlightthickness=0)
        self.sp_scrollbar = tk.Scrollbar(self.sp_outer_frame, orient="vertical", command=self.sp_canvas.yview)
        self.sp_scroll_frame = tk.Frame(self.sp_canvas, bg="#fcfcfc")
        self.sp_scroll_frame.bind("<Configure>", lambda e: self.sp_canvas.configure(scrollregion=self.sp_canvas.bbox("all")))
        self.sp_canvas.create_window((0, 0), window=self.sp_scroll_frame, anchor="nw")
        self.sp_canvas.configure(yscrollcommand=self.sp_scrollbar.set)
        self.sp_canvas.pack(side="left", fill="both", expand=True)
        self.sp_scrollbar.pack(side="right", fill="y")

        sp_btn_box = tk.Frame(self.sp_outer_frame, bg="white")
        sp_btn_box.pack(side="bottom", fill="x", pady=(5, 0))
        tk.Button(sp_btn_box, text="전체 선택", command=self.select_all_species).pack(side="left", padx=5)
        tk.Button(sp_btn_box, text="선택 해제", command=self.deselect_all_species).pack(side="left")

        # [3] Phase 1 경로
        self.p1_frame = ttk.LabelFrame(self.tab_main, text=" Phase 1: Baseline 모델 학습 ", padding=10)
        self.p1_frame.pack(fill="x", pady=5)

        p1_top = tk.Frame(self.p1_frame, bg="white")
        p1_top.pack(fill="x")
        tk.Checkbutton(
            p1_top,
            text="선택 종의 Baseline 모델 신규 학습 수행",
            variable=self.include_current,
            bg="white",
            fg="#e67e22",
            font=('Malgun Gothic', 9, 'bold'),
        ).pack(side="left")

        p1_path_frame = tk.Frame(self.p1_frame, bg="white")
        p1_path_frame.pack(fill="x", pady=5)
        tk.Label(p1_path_frame, text="학습 데이터 경로:", bg="white").pack(side="left")
        tk.Entry(p1_path_frame, textvariable=self.phase1_custom_path, font=('Malgun Gothic', 9)).pack(
            side="left", fill="x", expand=True, padx=5
        )
        tk.Button(
            p1_path_frame,
            text="찾아보기",
            cursor="hand2",
            command=lambda: self.browse_dir(self.phase1_custom_path),
        ).pack(side="left")

        # [4] Phase 2 시나리오
        self.s_outer_frame = ttk.LabelFrame(self.tab_main, text=" Phase 2: 학습 모델 Projection 구성 (02.ASCII_DATA 자동 인식) ", padding=10)
        self.s_outer_frame.pack(fill="both", expand=True, pady=5)

        chk_frame = tk.Frame(self.s_outer_frame, bg="white")
        chk_frame.pack(fill="x", pady=(0, 5))
        tk.Checkbutton(
            chk_frame,
            text="미래 시나리오 투영 및 통합연산 실행",
            variable=self.run_projection,
            bg="white",
            fg="#2980b9",
            font=('Malgun Gothic', 9, 'bold'),
        ).pack(side="left")

        self.scen_grid = tk.Frame(self.s_outer_frame, bg="white")
        self.scen_grid.pack(fill="both", expand=True)

        self.create_opt_tab()
        self.create_bottom_area()
        self.create_copyright_bar()

    def create_opt_tab(self):
        # Feature Selection
        f_frame = ttk.LabelFrame(self.tab_opt, text=" [1] Feature Selection ", padding=15)
        f_frame.pack(fill="x", pady=5)
        features = [
            ('Linear', 'linear'), ('Quadratic', 'quadratic'), ('Hinge', 'hinge'),
            ('Product', 'product'), ('Threshold', 'threshold'), ('Auto Feature', 'autofeature'),
        ]
        for i, (lbl, key) in enumerate(features):
            tk.Checkbutton(f_frame, text=lbl, variable=self.opt[key], bg="white").grid(row=0, column=i, padx=10)

        # 핵심 옵션
        v_frame = ttk.LabelFrame(self.tab_opt, text=" [2] 정규화 및 샘플링 설정 ", padding=15)
        v_frame.pack(fill="x", pady=5)
        tk.Label(v_frame, text="Beta Multiplier:", bg="white").grid(row=0, column=0, sticky="w")
        tk.Entry(v_frame, textvariable=self.opt['beta'], width=12).grid(row=0, column=1, padx=(5, 25), sticky="w")
        tk.Label(v_frame, text="Convergence:", bg="white").grid(row=0, column=2, sticky="w")
        tk.Entry(v_frame, textvariable=self.opt['convergence'], width=12).grid(row=0, column=3, padx=(5, 25), sticky="w")
        tk.Label(v_frame, text="Replicates:", bg="white").grid(row=0, column=4, sticky="w")
        tk.Entry(v_frame, textvariable=self.opt['reps'], width=12).grid(row=0, column=5, padx=(5, 25), sticky="w")

        tk.Label(v_frame, text="Rep Type:", bg="white").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            v_frame,
            textvariable=self.opt['rep_type'],
            values=["subsample", "bootstrap", "crossvalidate"],
            state="readonly",
            width=12,
        ).grid(row=1, column=1, padx=(5, 25), pady=(10, 0), sticky="w")

        tk.Label(v_frame, text="Max Iterations:", bg="white").grid(row=1, column=2, sticky="w", pady=(10, 0))
        tk.Entry(v_frame, textvariable=self.opt['max_iter'], width=12).grid(row=1, column=3, padx=(5, 25), pady=(10, 0), sticky="w")
        tk.Label(v_frame, text="Test %:", bg="white").grid(row=1, column=4, sticky="w", pady=(10, 0))
        tk.Entry(v_frame, textvariable=self.opt['test_pts'], width=12).grid(row=1, column=5, padx=(5, 25), pady=(10, 0), sticky="w")

        # 고급 출력
        o_frame = ttk.LabelFrame(self.tab_opt, text=" [3] 출력 및 고급 설정 ", padding=15)
        o_frame.pack(fill="x", pady=5)
        tk.Label(o_frame, text="Output Format:", bg="white").grid(row=0, column=0, sticky="w", pady=(0, 10))
        ttk.Combobox(
            o_frame,
            textvariable=self.opt['out_format'],
            values=["logistic", "cloglog", "raw"],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, padx=(5, 25), pady=(0, 10), sticky="w")

        checks = [
            ("Random Seed", "randomseed"), ("Response Curves", "response"),
            ("ROC Plots", "plots"), ("Jackknife", "jackknife"),
            ("Clamping", "clamping"), ("Extrapolate", "extrapolate"),
            ("Remove Duplicates", "removeduplicates"), ("Write Plot Data", "writeplot"),
        ]
        for i, (lbl, key) in enumerate(checks):
            tk.Checkbutton(o_frame, text=lbl, variable=self.opt[key], bg="white").grid(
                row=(i // 4) + 1, column=i % 4, sticky="w", pady=5, padx=2
            )

        # 고정 옵션 (시스템 무결성)
        fixed_sep = ttk.Separator(o_frame, orient="horizontal")
        fixed_sep.grid(row=4, column=0, columnspan=5, sticky="ew", pady=10)

        tk.Label(o_frame, text="시스템 고정 옵션:", bg="white", fg="#7f8c8d", font=('Malgun Gothic', 9, 'bold')).grid(
            row=5, column=0, columnspan=2, sticky="w"
        )

        tk.Checkbutton(o_frame, text="Output Grids", variable=self.opt['outputgrids'], state='disabled', bg="white", fg="#95a5a6").grid(
            row=5, column=2, sticky="w"
        )
        tk.Checkbutton(o_frame, text="Ask Overwrite", variable=self.opt['askoverwrite'], state='disabled', bg="white", fg="#95a5a6").grid(
            row=5, column=3, sticky="w"
        )

    def create_bottom_area(self):
        b_frame = tk.Frame(self.root, bg="#f8f9fa", padx=15, pady=5)
        b_frame.pack(fill="both", expand=True)

        log_container = tk.Frame(b_frame, bg="#1e1e1e")
        log_container.pack(fill="both", expand=True, pady=5)

        scrollbar = tk.Scrollbar(log_container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.log_txt = tk.Text(
            log_container,
            height=12,
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=('Consolas', 10),
            yscrollcommand=scrollbar.set,
        )
        self.log_txt.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log_txt.yview)

        self.p_bar = ttk.Progressbar(b_frame, orient="horizontal", mode="determinate")
        self.p_bar.pack(fill="x", pady=5)

        self.run_btn = tk.Button(
            b_frame,
            text="🚀 분석 시작",
            command=self.toggle_modeling,
            bg="#2c3e50",
            fg="white",
            font=('Malgun Gothic', 12, 'bold'),
            relief="flat",
        )
        self.run_btn.pack(fill="x", ipady=4, pady=3)

    def create_copyright_bar(self):
        footer = tk.Frame(self.root, bg="#f8f9fa")
        footer.pack(fill="x", side="bottom", padx=13, pady=5)
        tk.Label(
            footer,
            text="System: Python / MaxEnt 3.4.4",
            bg="#f8f9fa",
            fg="#bdc3c7",
            font=('Segoe UI', 8),
        ).pack(side="left")
        tk.Label(
            footer,
            text="© 2026. Chi-young Choi. All rights reserved.",
            bg="#f8f9fa",
            fg="#7f8c8d",
            font=('Segoe UI', 8),
        ).pack(side="right")

    def log(self, msg, level="INFO"):
        self.log_txt.config(state="normal")
        tag = "✔" if level == "SUCCESS" else "❌" if level == "ERROR" else "▶"
        self.log_txt.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {msg}\n")
        self.log_txt.see(tk.END)
        self.log_txt.config(state="disabled")

    def browse_base(self):
        path = filedialog.askdirectory()
        if path:
            self.base_dir.set(path)
            self.refresh_all_data()

    def browse(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def browse_dir(self, var):
        initial_dir = self.base_dir.get() if hasattr(self, 'base_dir') else "/"
        selected_path = filedialog.askdirectory(title="학습 데이터(.asc)가 포함된 폴더를 선택하세요", initialdir=initial_dir)
        if selected_path:
            var.set(os.path.normpath(selected_path))

    def select_all_species(self):
        for var in self.sp_vars.values():
            var.set(True)

    def deselect_all_species(self):
        for var in self.sp_vars.values():
            var.set(False)

    def refresh_all_data(self):
        # Species list
        sp_path = os.path.join(self.base_dir.get(), "01.SpeicesLIST")
        for widget in self.sp_scroll_frame.winfo_children():
            widget.destroy()
        self.sp_vars = {}
        if os.path.exists(sp_path):
            files = sorted([f.replace(".csv", "") for f in os.listdir(sp_path) if f.endswith(".csv")])
            for i, name in enumerate(files):
                v = tk.BooleanVar(value=True)
                self.sp_vars[name] = v
                tk.Checkbutton(self.sp_scroll_frame, text=name, variable=v, bg="#fcfcfc").grid(
                    row=i // 3, column=i % 3, sticky="w", padx=10
                )

        # Scenario scan (Future 전용: 02.ASCII_DATA/GCM-Future/{GCM}/{SSP}/{Period})
        ascii_path = os.path.join(self.base_dir.get(), "02.ASCII_DATA", "GCM-Future")
        for widget in self.scen_grid.winfo_children():

            widget.destroy()
        self.scen_vars = {"GCMs": {}, "SSPs": {}, "Periods": {}}
        if os.path.exists(ascii_path):
            valid_jobs = []            
            for g in os.listdir(ascii_path):
                gp = os.path.join(ascii_path, g)
                if os.path.isdir(gp):
                    for s in os.listdir(gp):
                        sp = os.path.join(gp, s)
                        if os.path.isdir(sp):
                            for p in os.listdir(sp):
                                valid_jobs.append((g, s, p))

            if valid_jobs:
                data_map = {
                    "GCMs": sorted(list(set(j[0] for j in valid_jobs))),
                    "SSPs": sorted(list(set(j[1] for j in valid_jobs))),
                    "Periods": sorted(list(set(j[2] for j in valid_jobs))),
                }
                for cat, items in data_map.items():
                    sub = tk.Frame(self.scen_grid, bg="#fcfcfc", bd=1, relief="solid", padx=5, pady=5)
                    sub.pack(side="left", fill="both", expand=True, padx=2)
                    tk.Label(sub, text=cat, font=('Segoe UI', 9, 'bold')).pack(anchor="w")
                    for item in items:
                        v = tk.BooleanVar(value=True)
                        self.scen_vars[cat][item] = v
                        tk.Checkbutton(sub, text=item, variable=v, bg="#fcfcfc").pack(anchor="w")

    def load_config(self):
        if not os.path.exists(self.config_file):
            return

        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')

        if 'SYSTEM' in config:
            s = config['SYSTEM']
            self.base_dir.set(s.get('base_dir', self.base_dir.get()))
            self.jar_path.set(s.get('jar_path', self.jar_path.get()))
            self.workers.set(s.getint('workers', self.workers.get()))
            self.java_threads.set(s.getint('java_threads', self.java_threads.get()))
            self.include_current.set(s.getboolean('include_current', self.include_current.get()))
            self.run_projection.set(s.getboolean('run_projection', self.run_projection.get()))
            if 'phase1_custom_path' in s:
                self.phase1_custom_path.set(s.get('phase1_custom_path'))

        if 'MAXENT_OPTIONS' in config:
            o = config['MAXENT_OPTIONS']
            for k, v in self.opt.items():
                if k not in o:
                    continue
                try:
                    if isinstance(v, tk.BooleanVar):
                        v.set(o.getboolean(k))
                    elif isinstance(v, tk.DoubleVar):
                        v.set(o.getfloat(k))
                    elif isinstance(v, tk.IntVar):
                        v.set(o.getint(k))
                    else:
                        v.set(o.get(k))
                except:
                    pass

    def save_config(self):
        config = configparser.ConfigParser()
        config['SYSTEM'] = {
            'base_dir': self.base_dir.get(),
            'jar_path': self.jar_path.get(),
            'workers': str(self.workers.get()),
            'java_threads': str(self.java_threads.get()),
            'include_current': str(self.include_current.get()),
            'run_projection': str(self.run_projection.get()),
            'phase1_custom_path': self.phase1_custom_path.get(),
        }
        config['MAXENT_OPTIONS'] = {k: str(v.get()) for k, v in self.opt.items()}
        with open(self.config_file, 'w', encoding='utf-8') as f:
            config.write(f)

    def toggle_modeling(self):
        state = self.run_state.get()

        if state == "IDLE":
            self.save_config()
            self.start_modeling()
        elif state == "RUNNING":
            # 실행 중 클릭 시: 일시정지
            self.stop_modeling()
        elif state == "PAUSED":
            # 일시정지 상태에서 클릭 시: 재시작(파이프라인 재실행)
            self.start_modeling()

    def _set_state(self, state):
        self.run_state.set(state)

        if state == "IDLE":
            self.run_btn.config(text="🚀 분석 시작", state="normal", bg="#2c3e50")
        elif state == "RUNNING":
            # 클릭하면 일시정지로 전환
            self.run_btn.config(text="⏸ 일시정지", state="normal", bg="#e67e22")
        elif state == "PAUSED":
            self.run_btn.config(text="↻ 재시작", state="normal", bg="#27ae60")
        elif state == "STOPPING":
            self.run_btn.config(text="⏳ 중단 중...", state="disabled", bg="#7f8c8d")

    def start_modeling(self):
        self.pause_requested = False
        self.is_running = True
        self._set_state("RUNNING")
        threading.Thread(target=self.process, daemon=True).start()

    def _terminate_pool_safely(self):
        p = self.pool
        self.pool = None
        if not p:
            return
        try:
            p.terminate()
        except:
            pass

    def stop_modeling(self):
        # '중단'이 아니라 '일시정지'로 동작 (PAUSED 상태 유지)
        self.pause_requested = True
        self._set_state("PAUSED")
        self.is_running = False
        self._terminate_pool_safely()
        self._log_threadsafe("일시정지됨 (↻ 재시작 버튼으로 다시 실행)", "INFO")

    # ------------------------------------------------------------
    # 3) process() 분할: Orchestrator + Step 메서드들
    # ------------------------------------------------------------

    def process(self):
        """오케스트레이터: 단계별 메서드를 순서대로 호출"""
        try:
            sp_list = self._collect_selected_species_or_abort()
            cfg, opts, target_reps = self._build_runtime_config()

            self._handle_replicates_mismatch_or_abort(sp_list, cfg, target_reps)

            trained_species, keep_existing_projection = self._run_phase1_train_all(sp_list, cfg, opts)

            if self.run_projection.get() and trained_species:
                sel_gcms, sel_ssps, sel_prds = self._collect_selected_scenarios()
                proj_jobs = self._build_projection_jobs(trained_species, sel_gcms, sel_ssps, sel_prds, cfg, target_reps, keep_existing_projection=keep_existing_projection)
                self._run_phase2_projection_all(proj_jobs, cfg, opts, target_reps)

                avg_jobs = self._build_averaging_jobs(trained_species, sel_gcms, sel_ssps, sel_prds, target_reps)
                self._run_phase3_averaging_all(avg_jobs, cfg, opts)

            self._on_pipeline_success()

        except Exception as e:
            # 사용자가 일시정지(또는 외부에서 java 종료)시: "Stopped by user" 예외는 오류 팝업 대신 PAUSED로 유지
            if isinstance(e, RuntimeError) and str(e) == "Stopped by user" and self.pause_requested:
                self._log_threadsafe("일시정지 처리됨", "INFO")
            else:
                self._on_pipeline_error(e)

        finally:
            self._restore_ui_state()

    # -------------------------
    # Step 0: 입력 수집/검증
    # -------------------------

    def _collect_selected_species_or_abort(self):
        sp_list = [name for name, var in self.sp_vars.items() if var.get()]
        if not sp_list:
            self.root.after(0, lambda: messagebox.showwarning("경고", "분석할 종을 선택해주세요."))
            self.is_running = False
            raise RuntimeError("No species selected")
        return sp_list

    def _build_runtime_config(self):
        cfg = {
            'base_dir': self.base_dir.get(),
            'jar_path': self.jar_path.get(),
            'memory': self.memory.get(),
            'java_threads': self.java_threads.get(),
            'phase1_custom_path': self.phase1_custom_path.get(),
        }

        # bool은 maxent에 문자열 true/false로 들어가므로 lower 처리
        opts = {k: (str(v.get()).lower() if isinstance(v.get(), bool) else v.get()) for k, v in self.opt.items()}
        target_reps = int(opts.get('reps', 1))
        return cfg, opts, target_reps

    # -------------------------
    # Step 0.5: reps 불일치 처리
    # -------------------------

    def _handle_replicates_mismatch_or_abort(self, sp_list, cfg, target_reps):
        mismatch_species_names = []
        mismatch_display_list = []

        for sp in sp_list:
            out_dir = os.path.normpath(os.path.join(cfg['base_dir'], "03.Results", sp, "GCM-Current", "SSP-Current", "Present"))
            if os.path.exists(out_dir):

                existing = [
                    f for f in os.listdir(out_dir)
                    if f.endswith(".lambdas") and os.path.getsize(os.path.join(out_dir, f)) > 0
                ]
                current_count = len(existing)
                if current_count > 0 and current_count != target_reps:
                    mismatch_species_names.append(sp)
                    mismatch_display_list.append(f"- {sp} (현재: {current_count}개 / 설정: {target_reps}개)")

        if not mismatch_display_list:
            return

        msg = "일부 종의 기존 학습 결과(Replicates)가 현재 설정과 다릅니다:\n\n"
        msg += "\n".join(mismatch_display_list[:10])
        if len(mismatch_display_list) > 10:
            msg += f"\n...외 {len(mismatch_display_list)-10}종"
        msg += "\n\n기존 파일을 삭제하고 새로 학습(재실행)하시겠습니까?\n('아니오'를 누르면 전체 분석이 중단됩니다.)"

        ok = messagebox.askyesno("Replicates 수 불일치 확인", msg)
        if not ok:
            self.is_running = False
            raise RuntimeError("User aborted due to replicate mismatch")

        # 삭제        for sp_name in mismatch_species_names:
            t_path = os.path.normpath(os.path.join(cfg['base_dir'], "03.Results", sp_name, "GCM-Current", "SSP-Current", "Present"))
            if os.path.exists(t_path):
                for f in os.listdir(t_path):
                    if f.lower().endswith((".lambdas", ".asc", ".html", ".csv", ".png", ".grd", ".xml")):
                        try:
                            os.remove(os.path.join(t_path, f))
                        except:
                            pass

    # -------------------------
    # Step 1: Phase 1
    # -------------------------

    def _run_phase1_train_all(self, sp_list, cfg, opts):
        self._log_threadsafe(f">>> Phase 1: 기준 모델({len(sp_list)}종) 병렬 학습 시작")

        trained_species = []
        needs_confirm = []
        keep_existing_projection = False
        self.root.after(0, lambda: self._progress_init(maximum=len(sp_list)))

        with Pool(processes=self.workers.get()) as pool:
            self.pool = pool
            try:
                for i, res in enumerate(pool.imap_unordered(run_phase1_train, [({'species': sp}, cfg, opts) for sp in sp_list]), 1):
                    self._abort_if_stopped()

                    sp_name = res['job']['species']
                    if res['status'] == "NEED_CONFIRM":
                        needs_confirm.append(res)
                        self._log_threadsafe(f"⚠️ Phase 1 NEED_CONFIRM(옵션 변경 감지): {sp_name}")
                    elif res['status'] in ("SUCCESS", "SKIPPED"):
                        trained_species.append(sp_name)
                        if res['status'] == "SKIPPED":
                            self._log_threadsafe(f"Phase 1 SKIPPED(기존 모델/메타 동일): {sp_name}")
                        else:
                            self._log_threadsafe(f"Phase 1 SUCCESS(신규 학습 완료): {sp_name}")
                    else:
                        err = res.get('error')
                        raise RuntimeError(f"종 '{sp_name}' 학습 중 오류 발생: {err}")

                    self.root.after(0, lambda v=i: self._progress_set(v))
            except Exception as e:
                # Pool terminate/사용자 일시정지 시, iterator가 예외로 끊기는 케이스를 정상 종료로 처리
                if self.pause_requested and not self.is_running:
                    raise RuntimeError("Stopped by user")
                raise
            finally:
                # with 블록이 close/join 하더라도, GUI 쪽에서는 더 이상 pool 참조하지 않게 정리
                if self.pool is pool:
                    self.pool = None

        # 옵션 변경으로 인해 재학습 여부 확인이 필요한 경우 사용자에게 선택을 받음
        if needs_confirm:
            def _fmt_val(v):
                if isinstance(v, bool):
                    return "true" if v else "false"
                if v is None:
                    return "-"
                return str(v)

            # 변경된 옵션 요약(상위 몇 개만 표시)
            diff_lines = []
            diff_pairs = []
            for r in needs_confirm:
                ex = r.get('existing_meta', {}) or {}
                old_fp = ex.get('fingerprint', {}) or {}
                new_fp = r.get('current_fingerprint', {}) or {}
                keys = sorted(set(old_fp.keys()) | set(new_fp.keys()))
                for k in keys:
                    ov = old_fp.get(k)
                    nv = new_fp.get(k)
                    if ov != nv:
                        diff_pairs.append((k, ov, nv))

            seen = set()
            for k, ov, nv in diff_pairs:
                if k in seen:
                    continue
                seen.add(k)
                diff_lines.append(f"- {k}: { _fmt_val(ov) } → { _fmt_val(nv) }")
                if len(diff_lines) >= 10:
                    break

            diff_text = ""
            if diff_lines:
                diff_text = "\n\n변경된 옵션(일부):\n" + "\n".join(diff_lines)
                if len(seen) < len({k for k, _, _ in diff_pairs}):
                    diff_text += "\n... (그 외 변경 항목 있음)"

            species_list = "\n".join([f"- {r['job']['species']}" for r in needs_confirm][:20])
            if len(needs_confirm) > 20:
                species_list += f"\n... 외 {len(needs_confirm)-20}종"

            msg = (
                "Phase 1 학습 결과(람다 파일)는 존재하지만, 학습 옵션이 이전과 달라졌습니다.\n\n"
                f"대상 종:\n{species_list}\n\n"
                f"{diff_text}\n\n"
                "재학습을 진행하시겠습니까?\n"
                "예(Yes): Phase 1을 다시 학습(기존 Current 결과 폴더를 삭제 후 재학습)\n"
                "아니오(No): 기존 모델을 그대로 사용하고 Phase 2로 진행"
            )

            ok = messagebox.askyesno("Phase 1 옵션 변경 감지", msg)
            if ok:
                # 사용자가 재학습을 선택한 경우
                self._log_threadsafe("Phase 1 결정: 재학습(Yes) 선택", "INFO")
                import shutil
                for r in needs_confirm:
                    sp = r['job']['species']
                    train_dir = os.path.join(cfg['base_dir'], "03.Results", sp, "GCM-Current", "SSP-Current", "Present")
                    shutil.rmtree(train_dir, ignore_errors=True)
                    os.makedirs(train_dir, exist_ok=True)

                self._log_threadsafe(f">>> Phase 1 재학습 선택: {len(needs_confirm)}종 재학습 시작")

                # 재학습은 순차(안전)로 처리: GUI 중단/일시정지 시 안전하게 끊기게 함
                for r in needs_confirm:
                    self._abort_if_stopped()
                    sp = r['job']['species']
                    res2 = run_phase1_train(({'species': sp}, cfg, opts))
                    if res2['status'] != "SUCCESS":
                        err = res2.get('error')
                        raise RuntimeError(f"종 '{sp}' 재학습 중 오류 발생: {err}")
                    if sp not in trained_species:
                        trained_species.append(sp)
                    self._log_threadsafe(f"Phase 1 RETRAIN SUCCESS: {sp}")
            else:
                # No 선택 시: 기존 모델을 그대로 사용하므로 Phase2 대상으로 포함
                self._log_threadsafe("Phase 1 결정: 기존 모델 사용(No) 선택", "INFO")
                keep_existing_projection = True
                for r in needs_confirm:
                    sp = r['job']['species']
                    if sp not in trained_species:
                        trained_species.append(sp)
                self._log_threadsafe(f">>> Phase 1 재학습 건너뜀 선택: 기존 모델로 Phase 2 진행 ({len(needs_confirm)}종)")

        return trained_species, keep_existing_projection

    # -------------------------
    # Step 1.5: Phase 1 메타
    # -------------------------

    def _update_train_fingerprint(self, trained_species, target_reps):
        impact_keys = [
            'linear', 'quadratic', 'hinge', 'product', 'threshold', 'autofeature',
            'beta', 'convergence', 'reps', 'rep_type', 'test_pts', 'max_iter',
            'clamping', 'extrapolate',
        ]
        current_fp = {k: self.opt[k].get() for k in impact_keys if k in self.opt}

        def _compute_model_id(train_dir, species, reps):
            h = hashlib.sha256()
            if reps > 1:
                lambda_files = [os.path.join(train_dir, f"{species}_{i}.lambdas") for i in range(reps)]
            else:
                lambda_files = [os.path.join(train_dir, f"{species}.lambdas")]

            for fp in lambda_files:
                if not os.path.exists(fp):
                    return None
                # 파일 내용 기반 해시(재학습/결과 차이 감지용)
                with open(fp, 'rb') as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)

            return h.hexdigest()

        for sp in trained_species:
            train_dir = os.path.join(self.base_dir.get(), "03.Results", sp, "GCM-Current", "SSP-Current", "Present")
            os.makedirs(train_dir, exist_ok=True)

            origin_meta_path = os.path.join(train_dir, "model_meta.json")

            model_id = _compute_model_id(train_dir, sp, target_reps)

            existing_meta = None
            if os.path.exists(origin_meta_path):
                try:
                    with open(origin_meta_path, 'r', encoding='utf-8') as f:
                        existing_meta = json.load(f)
                except:
                    existing_meta = None

            if existing_meta:
                same_fp = (existing_meta.get('fingerprint') == current_fp)
                same_reps = (existing_meta.get('target_reps') == target_reps)
                same_model = (existing_meta.get('model_id') == model_id) and (model_id is not None)
                if same_fp and same_reps and same_model:
                    continue

            train_meta = {
                "species": sp,
                "fingerprint": current_fp,
                "target_reps": target_reps,
                "model_id": model_id,
                # NOTE: train_time은 비교 로직에서 제외하거나 별도 키로 관리하는 편이 안전합니다.
                "train_time": time.time(),
            }
            with open(origin_meta_path, 'w', encoding='utf-8') as f:
                json.dump(train_meta, f, indent=4)

    # -------------------------
    # Step 2: Phase 2
    # -------------------------

    def _collect_selected_scenarios(self):
        sel_gcms = [k for k, v in self.scen_vars['GCMs'].items() if v.get()]
        sel_ssps = [k for k, v in self.scen_vars['SSPs'].items() if v.get()]
        sel_prds = [k for k, v in self.scen_vars['Periods'].items() if v.get()]
        return sel_gcms, sel_ssps, sel_prds

    def _build_projection_jobs(self, trained_species, sel_gcms, sel_ssps, sel_prds, cfg, target_reps, keep_existing_projection=False):
        proj_jobs = []
        for sp in trained_species:
            train_dir = os.path.join(cfg['base_dir'], "03.Results", sp, "GCM-Current", "SSP-Current", "Present")
            origin_meta_path = os.path.join(train_dir, "model_meta.json")
            with open(origin_meta_path, 'r', encoding='utf-8') as f:
                t_meta = json.load(f)

            for g in sel_gcms:
                for s in sel_ssps:
                    for p in sel_prds:
                        env_path = os.path.join(cfg['base_dir'], "02.ASCII_DATA", "GCM-Future", g, s, p)
                        out_dir = os.path.join(cfg['base_dir'], "03.Results", sp, "GCM-Future", g, s, p)
                        local_meta_path = os.path.join(out_dir, "model_meta.json")


                        should_reset = False
                        if not keep_existing_projection:
                            if os.path.exists(local_meta_path):
                                try:
                                    with open(local_meta_path, 'r', encoding='utf-8') as f:
                                        l_meta = json.load(f)

                                    # 무결성 비교: 옵션 + replicates + 학습결과(model_id)
                                    same_fp = (t_meta.get('fingerprint') == l_meta.get('fingerprint'))
                                    same_reps = (t_meta.get('target_reps') == l_meta.get('target_reps'))
                                    same_model = (t_meta.get('model_id') == l_meta.get('model_id')) and (t_meta.get('model_id') is not None)
                                    if not (same_fp and same_reps and same_model):
                                        should_reset = True
                                except:
                                    should_reset = True
                            else:
                                if os.path.exists(out_dir):
                                    should_reset = True
                        else:
                            # "기존 모델 사용" 모드: 옵션 변경과 상관없이 기존 투영 결과 폴더를 유지(삭제 금지)
                            should_reset = False

                        if should_reset:
                            import shutil
                            shutil.rmtree(out_dir, ignore_errors=True)
                            os.makedirs(out_dir, exist_ok=True)
                            with open(local_meta_path, 'w', encoding='utf-8') as f:
                                json.dump(t_meta, f, indent=4)
                        else:
                            os.makedirs(out_dir, exist_ok=True)
                            # meta가 없다면 1회 기록(이후 무결성 비교/추적용)
                            if not os.path.exists(local_meta_path):
                                try:
                                    with open(local_meta_path, 'w', encoding='utf-8') as f:
                                        json.dump(t_meta, f, indent=4)
                                except:
                                    pass

                        for r in range(target_reps):
                            target_file = os.path.join(out_dir, f"{sp}_{r}.asc")
                            # 누락(또는 0바이트)만 복구하도록 job 생성 (있으면 Phase2에서 SKIP됨)
                            if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                                continue
                            proj_jobs.append({
                                'species': sp, 'gcm': g, 'ssp': s, 'period': p, 'rep_index': r,
                                'folder_key': f"[{sp}] {g} / {s} / {p}",
                                'lambda_path': os.path.join(train_dir, f"{sp}_{r}.lambdas"),
                                'proj_layers': env_path,
                                'out_dir': out_dir,
                                'target_file': target_file,
                                't_meta': t_meta,
                            })

        return proj_jobs

    def _run_phase2_projection_all(self, proj_jobs, cfg, opts, target_reps):
        if not proj_jobs:
            self._log_threadsafe(">>> Phase 2: 투영할 작업이 없습니다(모두 완료되어 SKIP).")
            return

        self._log_threadsafe(">>> Phase 2: 시나리오별 리플리케이트 병렬 투영 시작")

        total_jobs = len(proj_jobs)
        self.root.after(0, lambda: self._progress_init(maximum=total_jobs))

        folder_stats = {}
        started_folders = set()
        total_failed = 0
        # env_path(폴더) 단위 락 준비: 같은 env_path는 1개씩만 실행
        if self._mgr is None:
            self._mgr = Manager()
        lock_map = self._mgr.dict()
        for j in proj_jobs:
            ep = os.path.normpath(j.get('proj_layers', ''))
            if not ep:
                continue
            if ep not in lock_map:
                lock_map[ep] = self._mgr.Lock()

        # 각 job에 lock_map 참조 주입
        for j in proj_jobs:
            j['_lock_map'] = lock_map

        with Pool(processes=self.workers.get()) as pool:
            self.pool = pool
            try:
                for i, res in enumerate(pool.imap_unordered(run_phase2_projection, [(j, cfg, opts) for j in proj_jobs]), 1):
                    self._abort_if_stopped()

                    job = res['job']
                    f_key = job['folder_key']

                    if f_key not in started_folders:
                        started_folders.add(f_key)
                        self._log_threadsafe(f"Phase 2 START: {f_key}")

                    folder_stats.setdefault(f_key, {'success': 0, 'skipped': 0, 'failed': 0})

                    status = res['status']
                    if status == "SUCCESS":
                        folder_stats[f_key]['success'] += 1
                    elif status == "SKIPPED":
                        folder_stats[f_key]['skipped'] += 1
                    else:
                        folder_stats[f_key]['failed'] += 1
                        total_failed += 1
                        self._log_threadsafe(f"❌ 실패: {job['species']}_Rep{job['rep_index']} | {res.get('error')}")

                    done_count = sum(folder_stats[f_key].values())
                    if done_count == target_reps:
                        out_dir = job['out_dir']
                        with open(os.path.join(out_dir, "model_meta.json"), 'w', encoding='utf-8') as f:
                            json.dump(job['t_meta'], f, indent=4)

                        s = folder_stats[f_key]
                        self._log_threadsafe(
                            f"Phase 2 DONE: {f_key} | SUCCESS={s['success']} SKIPPED={s['skipped']} FAILED={s['failed']}"
                        )

                    self.root.after(0, lambda v=i: self._progress_set(v))
            except Exception as e:
                if self.pause_requested and not self.is_running:
                    raise RuntimeError("Stopped by user")
                raise
            finally:
                if self.pool is pool:
                    self.pool = None

        self._log_threadsafe(">>> Phase 2: 시나리오 투영 공정 종료")
        if total_failed > 0:
            raise RuntimeError(f"Phase 2 투영 실패: {total_failed}건")

    # -------------------------
    # Step 3: Phase 3
    # -------------------------

    def _build_averaging_jobs(self, trained_species, sel_gcms, sel_ssps, sel_prds, target_reps):
        avg_jobs = []
        for sp in trained_species:
            for g in sel_gcms:
                for s in sel_ssps:
                    for p in sel_prds:
                        avg_jobs.append({'species': sp, 'gcm': g, 'ssp': s, 'period': p, 'target_reps': target_reps})
        return avg_jobs

    def _run_phase3_averaging_all(self, avg_jobs, cfg, opts):
        if not avg_jobs:
            return

        self._log_threadsafe(">>> Phase 3: 리플리케이트 결과 통합 및 통계 세트 생성 시작")

        total_avg = len(avg_jobs)
        self.root.after(0, lambda: self._progress_init(maximum=total_avg))

        total_failed = 0

        with Pool(processes=self.workers.get()) as pool:
            self.pool = pool
            try:
                for i, res in enumerate(pool.imap_unordered(run_phase3_averaging, [(j, cfg, opts) for j in avg_jobs]), 1):
                    self._abort_if_stopped()

                    status = res.get('status')
                    if status not in ("SUCCESS", "SKIPPED"):
                        total_failed += 1
                        job = res.get('job', {})
                        self._log_threadsafe(f"❌ 실패: {job.get('species')} | {job.get('gcm')} | {job.get('ssp')} | {job.get('period')} | {res.get('error')}")
                    else:
                        job = res.get('job', {})
                        mode = res.get('mode', '-')
                        if status == "SUCCESS":
                            self._log_threadsafe(f"Phase 3 SUCCESS({mode}): {job.get('species')} | {job.get('gcm')} | {job.get('ssp')} | {job.get('period')}")
                        else:
                            self._log_threadsafe(f"Phase 3 SKIPPED({mode}): {job.get('species')} | {job.get('gcm')} | {job.get('ssp')} | {job.get('period')}")

                    self.root.after(0, lambda v=i: self._progress_set(v))
            except Exception as e:
                if self.pause_requested and not self.is_running:
                    raise RuntimeError("Stopped by user")
                raise
            finally:
                if self.pool is pool:
                    self.pool = None

        self._log_threadsafe(">>> Phase 3: 모든 시나리오 통합 공정 종료")
        if total_failed > 0:
            raise RuntimeError(f"Phase 3 통합 실패: {total_failed}건")

    # -------------------------
    # 공통: 중단/로그/UI
    # -------------------------

    def _abort_if_stopped(self):
        if not self.is_running:
            raise RuntimeError("Stopped by user")

    def _log_threadsafe(self, msg, level="INFO"):
        # Tkinter 위젯 접근은 after로 메인스레드에서 실행
        self.root.after(0, lambda: self.log(msg, level))

    def _progress_init(self, maximum):
        self.p_bar['maximum'] = maximum
        self.p_bar['value'] = 0

    def _progress_set(self, value):
        self.p_bar['value'] = value

    def _on_pipeline_success(self):
        self.root.after(0, lambda: messagebox.showinfo("완료", "학습, 투영 및 통계 통합이 모두 성공적으로 종료되었습니다."))

    def _on_pipeline_error(self, e):
        self.is_running = False
        self.root.after(0, lambda: messagebox.showerror("오류", str(e)))

    def _restore_ui_state(self):
        self.is_running = False
        # 일시정지 상태라면 IDLE로 강제 복귀하지 않음 (버튼/상태 유지)
        if self.run_state.get() == "PAUSED":
            return
        self.root.after(0, lambda: self._set_state("IDLE"))


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    freeze_support()
    root = tk.Tk()
    app = MaxEntFullGUI(root)
    root.mainloop()