#!/usr/bin/env python3
"""
TensorBoard 로그 파일 읽기 및 내보내기 스크립트
학습 중 기록된 메트릭들을 추출하고 LLM 분석용으로 내보냅니다.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from datetime import datetime

try:
    from tensorboard.backend.event_processing import event_accumulator
    from collections import defaultdict
    import numpy as np
except ImportError:
    print("❌ tensorboard 패키지가 필요합니다.")
    print("설치: pip install tensorboard")
    exit(1)


def read_tensorboard_events(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    TensorBoard 로그 디렉토리에서 모든 이벤트를 읽어옵니다.
    
    Args:
        log_dir: TensorBoard 로그 디렉토리 경로
        
    Returns:
        메트릭 이름을 키로, (step, value) 튜플 리스트를 값으로 하는 딕셔너리
    """
    # 모든 이벤트 파일 찾기
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"⚠️  {log_dir}에서 이벤트 파일을 찾을 수 없습니다.")
        return {}
    
    print(f"📁 발견된 이벤트 파일: {len(event_files)}개")
    
    # 모든 메트릭 데이터를 수집
    all_metrics = defaultdict(list)
    
    for event_file in sorted(event_files):
        print(f"   읽는 중: {os.path.basename(event_file)}")
        
        try:
            # EventAccumulator 초기화
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # 사용 가능한 태그(메트릭) 확인
            tags = ea.Tags()
            
            # 스칼라 값 읽기
            for tag in tags['scalars']:
                events = ea.Scalars(tag)
                for event in events:
                    all_metrics[tag].append((event.step, event.value))
                    
        except Exception as e:
            print(f"   ⚠️  파일 읽기 실패: {e}")
            continue
    
    # 각 메트릭을 step 순서로 정렬
    for tag in all_metrics:
        all_metrics[tag] = sorted(all_metrics[tag], key=lambda x: x[0])
    
    return dict(all_metrics)


def export_to_json(metrics: Dict[str, List[Tuple[int, float]]], output_path: str):
    """메트릭을 JSON 파일로 내보냅니다."""
    
    data = {
        "export_time": datetime.now().isoformat(),
        "metrics": {}
    }
    
    for tag, values in metrics.items():
        if not values:
            continue
        
        steps, vals = zip(*values)
        
        data["metrics"][tag] = {
            "data_points": [{"step": s, "value": v} for s, v in values],
            "summary": {
                "count": len(values),
                "min_step": min(steps),
                "max_step": max(steps),
                "min_value": float(min(vals)),
                "max_value": float(max(vals)),
                "mean_value": float(np.mean(vals)),
                "final_value": float(vals[-1]),
                "final_step": int(steps[-1])
            }
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 파일 저장: {output_path}")


def export_to_markdown(metrics: Dict[str, List[Tuple[int, float]]], output_path: str, run_name: str = "Training Run"):
    """메트릭을 Markdown 파일로 내보냅니다 (LLM 분석용)."""
    
    lines = []
    lines.append(f"# TensorBoard 메트릭 분석 리포트")
    lines.append(f"\n**Run Name:** {run_name}")
    lines.append(f"**Export Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Metrics:** {len(metrics)}")
    lines.append("\n---\n")
    
    for tag, values in sorted(metrics.items()):
        if not values:
            continue
        
        steps, vals = zip(*values)
        
        lines.append(f"## 📊 {tag}\n")
        lines.append(f"- **데이터 포인트 수:** {len(values)}")
        lines.append(f"- **Step 범위:** {min(steps)} ~ {max(steps)}")
        lines.append(f"- **최소값:** {min(vals):.6f} (step {steps[vals.index(min(vals))]})")
        lines.append(f"- **최대값:** {max(vals):.6f} (step {steps[vals.index(max(vals))]})")
        lines.append(f"- **평균값:** {np.mean(vals):.6f}")
        lines.append(f"- **최종값:** {vals[-1]:.6f} (step {steps[-1]})")
        
        # 전체 데이터 테이블
        lines.append(f"\n### 전체 데이터 (Step별)\n")
        lines.append("| Step | Value |")
        lines.append("|------|-------|")
        for step, val in values:
            lines.append(f"| {step} | {val:.6f} |")
        
        lines.append("\n---\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Markdown 파일 저장: {output_path}")


def export_to_csv(metrics: Dict[str, List[Tuple[int, float]]], output_path: str):
    """메트릭을 CSV 파일로 내보냅니다."""
    
    lines = []
    
    # 헤더
    metric_names = sorted(metrics.keys())
    lines.append("step," + ",".join(metric_names))
    
    # 모든 step 수집
    all_steps = set()
    for values in metrics.values():
        all_steps.update(s for s, _ in values)
    
    # step별 데이터 매핑
    step_data = {step: {} for step in sorted(all_steps)}
    for tag, values in metrics.items():
        for step, val in values:
            step_data[step][tag] = val
    
    # CSV 작성
    for step in sorted(step_data.keys()):
        row = [str(step)]
        for metric in metric_names:
            val = step_data[step].get(metric, "")
            row.append(f"{val:.6f}" if val != "" else "")
        lines.append(",".join(row))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ CSV 파일 저장: {output_path}")


def print_metric_summary(metrics: Dict[str, List[Tuple[int, float]]]):
    """메트릭 요약 정보를 출력합니다."""
    
    if not metrics:
        print("❌ 메트릭 데이터가 없습니다.")
        return
    
    print("\n" + "="*80)
    print("📊 TensorBoard 메트릭 요약")
    print("="*80)
    
    for tag, values in sorted(metrics.items()):
        if not values:
            continue
            
        steps, vals = zip(*values)
        
        print(f"\n📈 {tag}")
        print(f"   총 데이터 포인트: {len(values)}개")
        print(f"   Step 범위: {min(steps)} ~ {max(steps)}")
        print(f"   최소값: {min(vals):.6f} (step {steps[vals.index(min(vals))]})") 
        print(f"   최대값: {max(vals):.6f} (step {steps[vals.index(max(vals))]})")
        print(f"   평균값: {np.mean(vals):.6f}")
        print(f"   최종값: {vals[-1]:.6f} (step {steps[-1]})")
        
        # 최근 10개 데이터 포인트 표시
        print(f"   최근 10개 데이터:")
        for step, val in values[-10:]:
            print(f"      Step {step:4d}: {val:.6f}")


def compare_runs(log_dirs: List[str]):
    """여러 실행의 결과를 비교합니다. (콘솔 출력용)"""
    
    print("\n" + "="*80)
    print("🔄 여러 실행 비교")
    print("="*80)
    
    all_runs = {}
    
    for log_dir in log_dirs:
        run_name = os.path.basename(log_dir)
        print(f"\n📂 {run_name} 읽는 중...")
        metrics = read_tensorboard_events(log_dir)
        all_runs[run_name] = metrics
    
    # 공통 메트릭 찾기
    common_metrics = set.intersection(*[set(m.keys()) for m in all_runs.values()])
    
    if not common_metrics:
        print("\n⚠️  공통 메트릭이 없습니다.")
        return
    
    print(f"\n📊 공통 메트릭 ({len(common_metrics)}개):")
    
    for metric in sorted(common_metrics):
        print(f"\n📈 {metric}")
        print(f"   {'Run':<30} {'최종값':<15} {'최소값':<15} {'최대값':<15}")
        print(f"   {'-'*75}")
        
        for run_name, metrics in all_runs.items():
            if metric in metrics and metrics[metric]:
                vals = [v for _, v in metrics[metric]]
                final_val = vals[-1]
                min_val = min(vals)
                max_val = max(vals)
                
                print(f"   {run_name:<30} {final_val:<15.6f} {min_val:<15.6f} {max_val:<15.6f}")


def main():
    """메인 함수"""
    
    workspace = Path("/workspace/MiniR1_sogmgm_ver")
    output_dir = workspace / "logs" / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 사용 가능한 TensorBoard 로그 디렉토리 찾기
    log_dirs = [
        workspace / "logs" / "tensorboard_1.7b",
        workspace / "logs" / "tensorboard_4b_lora",
    ]
    
    existing_dirs = [d for d in log_dirs if d.exists()]
    
    if not existing_dirs:
        print("❌ TensorBoard 로그 디렉토리를 찾을 수 없습니다.")
        return
    
    print("🔍 발견된 TensorBoard 로그:")
    for i, log_dir in enumerate(existing_dirs, 1):
        print(f"   {i}. {log_dir}")
    
    # 각 디렉토리의 메트릭 읽기 및 내보내기
    all_runs_data = {}
    
    for log_dir in existing_dirs:
        print(f"\n{'='*80}")
        print(f"📊 {log_dir.name} 분석 중...")
        print(f"{'='*80}")
        
        metrics = read_tensorboard_events(str(log_dir))
        
        if not metrics:
            print(f"⚠️  {log_dir.name}에서 메트릭을 읽을 수 없습니다.")
            continue
        
        print_metric_summary(metrics)
        
        # 파일명에 사용할 run name
        run_name = log_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON으로 내보내기
        json_path = output_dir / f"{run_name}_{timestamp}.json"
        export_to_json(metrics, str(json_path))
        
        # Markdown으로 내보내기 (LLM 분석용)
        md_path = output_dir / f"{run_name}_{timestamp}.md"
        export_to_markdown(metrics, str(md_path), run_name)
        
        # CSV로 내보내기
        csv_path = output_dir / f"{run_name}_{timestamp}.csv"
        export_to_csv(metrics, str(csv_path))
        
        all_runs_data[run_name] = metrics
    
    # 여러 실행 비교 (있는 경우)
    if len(all_runs_data) > 1:
        print(f"\n{'='*80}")
        print("🔄 여러 실행 비교")
        print(f"{'='*80}")
        
        # 비교 리포트 생성
        comparison_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        create_comparison_report(all_runs_data, str(comparison_path))
    
    print("\n" + "="*80)
    print(f"✅ 분석 완료! 결과는 {output_dir}에 저장되었습니다.")
    print("="*80)
    print(f"\n📁 생성된 파일들:")
    for file in sorted(output_dir.glob("*")):
        print(f"   - {file.name}")


def create_comparison_report(all_runs: Dict[str, Dict[str, List[Tuple[int, float]]]], output_path: str):
    """여러 실행을 비교하는 Markdown 리포트를 생성합니다."""
    
    lines = []
    lines.append("# 학습 실행 비교 리포트")
    lines.append(f"\n**생성 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**비교 대상:** {', '.join(all_runs.keys())}")
    lines.append("\n---\n")
    
    # 공통 메트릭 찾기
    common_metrics = set.intersection(*[set(m.keys()) for m in all_runs.values()])
    
    if not common_metrics:
        lines.append("⚠️ 공통 메트릭이 없습니다.\n")
    else:
        lines.append(f"## 공통 메트릭 비교 ({len(common_metrics)}개)\n")
        
        for metric in sorted(common_metrics):
            lines.append(f"### 📊 {metric}\n")
            lines.append("| Run | 최종값 | 최소값 | 최대값 | 평균값 | Step 범위 |")
            lines.append("|-----|--------|--------|--------|--------|-----------|")
            
            for run_name, metrics in all_runs.items():
                if metric in metrics and metrics[metric]:
                    steps, vals = zip(*metrics[metric])
                    final_val = vals[-1]
                    min_val = min(vals)
                    max_val = max(vals)
                    mean_val = np.mean(vals)
                    step_range = f"{min(steps)}-{max(steps)}"
                    
                    lines.append(f"| {run_name} | {final_val:.6f} | {min_val:.6f} | {max_val:.6f} | {mean_val:.6f} | {step_range} |")
            
            lines.append("")
    
    # 각 run별 고유 메트릭
    lines.append("## Run별 고유 메트릭\n")
    for run_name, metrics in all_runs.items():
        other_metrics = set()
        for other_run, other_m in all_runs.items():
            if other_run != run_name:
                other_metrics.update(other_m.keys())
        
        unique = set(metrics.keys()) - other_metrics
        if unique:
            lines.append(f"### {run_name}")
            for m in sorted(unique):
                lines.append(f"- {m}")
            lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ 비교 리포트 저장: {output_path}")


if __name__ == "__main__":
    main()
