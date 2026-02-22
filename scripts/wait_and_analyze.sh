#!/bin/bash
# Wait for ablation experiments to complete, then run full analysis + plots
ABLATION_PID=99732
LOG=/Users/david/jepa-substrate/results/ablation.log
echo "Waiting for ablation experiments (PID $ABLATION_PID)..."

while kill -0 $ABLATION_PID 2>/dev/null; do
    DONE=$(grep -c "DONE\|SKIP" "$LOG" 2>/dev/null || echo "0")
    echo "$(date +%H:%M:%S) | $DONE/60 done"
    sleep 300
done

echo "Ablation experiments finished!"
echo "Running full analysis..."

cd /Users/david/jepa-substrate
.venv/bin/python3 scripts/run_analysis.py 2>&1
echo "Running plots..."
.venv/bin/python3 scripts/generate_plots.py 2>&1
echo "ALL DONE."
