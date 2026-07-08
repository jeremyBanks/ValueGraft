#!/usr/bin/env bash
cd /Users/jeb/experimentation
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
ENTRIES="1:Qwen/Qwen3-30B-A3B-Instruct-2507:24 2:Qwen/Qwen3-32B:24 3:Qwen/Qwen2.5-32B-Instruct:24 4:google/gemma-4-31B-it:24 5:google/gemma-4-26B-A4B-it:24 6:mistralai/Mixtral-8x7B-Instruct-v0.1:12 7:mistralai/Mistral-Small-24B-Instruct-2501:12 8:google/gemma-3-27b-it:12 9:allenai/OLMo-2-0325-32B-Instruct:12 10:Qwen/Qwen3.6-35B-A3B:12 11:Qwen/Qwen3.6-27B:12 12:zai-org/GLM-4-32B-0414:12 13:openai/gpt-oss-20b:12 14:microsoft/phi-4:12 15:01-ai/Yi-1.5-34B-Chat:12 16:nvidia/Llama-3_3-Nemotron-Super-49B-v1_5:12"
for e in $ENTRIES; do
  i=${e%%:*}; rest=${e#*:}; M=${rest%:*}; C=${rest##*:}
  [ -f ".pod_w${i}_state.json" ] && grep -q '"id"' ".pod_w${i}_state.json" 2>/dev/null || continue
  echo "repairing w$i ($M, $C convs)"
  MODELS="$M" SC_CONV_LIMIT=$C SC_POD_DISK=400 bash scripts/launch_pod.sh "w$i" scripts/job_sweep.sh > "$S/launch_w$i.log" 2>&1 &
  sleep 10
done
wait
echo "repair pass done"
