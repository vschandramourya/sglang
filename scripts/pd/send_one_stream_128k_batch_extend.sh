#!/usr/bin/env bash
set -euo pipefail

HOST="http://10.173.2.72:8091"

while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

export HOST
export MODEL="cursor/dsv31-gb200-tgl-test-02"
export BASE_CONTEXT="$(cat <<'EOF'
New York City is composed of five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. Each borough has distinct cultural, historical, and social characteristics. Manhattan is often associated with finance, Broadway, museums, and iconic landmarks. Brooklyn is known for creative communities, food scenes, and waterfront parks. Queens is one of the most ethnically diverse places in the world. The Bronx is the birthplace of hip-hop and home to important sports and cultural institutions. Staten Island offers suburban characteristics and views of Manhattan Harbor.

Tourism in New York City generally clusters around several categories:
1. Iconic landmarks and skyline experiences
2. Cultural institutions such as museums and theaters
3. Neighborhood exploration and food

Visitors often have limited time and want to maximize impact. Therefore, answers should prioritize experiences that are:
- High cultural or historical significance
- Widely regarded as representative of New York City
- Accessible to first-time visitors

When ranking or selecting activities, consider factors such as:
- Global recognition
- Density of experiences
- Time efficiency
- Seasonal neutrality

Examples of iconic landmarks include (but are not limited to): Central Park, Times Square, the Statue of Liberty, Empire State Building, Brooklyn Bridge, and One World Trade Center.
Examples of cultural institutions include the Metropolitan Museum of Art, Museum of Modern Art, Broadway theaters, and Lincoln Center.
Examples of neighborhood experiences include walking through SoHo, Greenwich Village, Chinatown, Harlem, Williamsburg, and Dumbo.

Food is an important but optional dimension. If included, it should be representative (e.g., New York–style pizza, bagels, deli culture) rather than overly niche.

EOF
)"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

NUM_REQUESTS=10
python3 - "$TMPDIR" "$NUM_REQUESTS" <<'PY'
import json, os, sys

tmpdir = sys.argv[1]
num_requests = int(sys.argv[2])

base_context = os.environ["BASE_CONTEXT"]
target_tokens = 60 * 1024
chars_per_token = 4
target_chars = target_tokens * chars_per_token

repetitions = (target_chars // len(base_context)) + 1
expanded_context = (base_context + "\n") * repetitions
expanded_context = expanded_context[:target_chars]

system_prompt = f"""You are a highly knowledgeable travel expert and city historian with deep familiarity with New York City across culture, history, food, architecture, arts, transportation, and neighborhoods.
Your task is to answer questions about New York City with clarity, depth, and structured reasoning.

Before answering, carefully consider the following background context and constraints. This context is intentionally long and should be treated as important reference material.

{expanded_context}

Tone requirements:
- Clear and confident
- Informative but concise
- Structured with numbered lists when appropriate"""

user_question_cores = [
    "What are the top 3 things to do in New York City for a first-time visitor? List exactly three items with a short title and 2-3 sentence explanation each.",
    "What are the best museums to visit in Manhattan? List exactly three items with a short title and 2-3 sentence explanation each.",
    "Where can I find the best pizza in New York City? List exactly three neighborhoods with a short title and 2-3 sentence explanation each.",
    "What are the most scenic walking routes in New York City? List exactly three routes with a short title and 2-3 sentence explanation each.",
    "What are the best free activities in New York City? List exactly three items with a short title and 2-3 sentence explanation each.",
    "What neighborhoods should I explore in Brooklyn? List exactly three neighborhoods with a short title and 2-3 sentence explanation each.",
    "What are the best viewpoints to see the NYC skyline? List exactly three locations with a short title and 2-3 sentence explanation each.",
    "What Broadway shows should a first-timer see? List exactly three recommendations with a short title and 2-3 sentence explanation each.",
    "What are the best parks in New York City besides Central Park? List exactly three parks with a short title and 2-3 sentence explanation each.",
    "What historical landmarks should I visit in NYC? List exactly three landmarks with a short title and 2-3 sentence explanation each.",
]

user_target_tokens = 5 * 1024
user_target_chars = user_target_tokens * chars_per_token

user_questions = []
for i, q in enumerate(user_question_cores):
    padding_base = f"[Context block {i+1}] " + base_context
    padding_reps = (user_target_chars // len(padding_base)) + 1
    padding = (padding_base + "\n") * padding_reps
    padding = padding[:user_target_chars - len(q) - 50]
    user_questions.append(f"{padding}\n\nNow answer this question:\n{q}")

initial_payload = {
    "model": os.environ["MODEL"],
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hello, I'm ready to ask you questions about New York City."},
    ],
    "stream": True,
    "stream_options": {"include_usage": True},
    "temperature": 0.0,
    "max_tokens": 1,
}
with open(f"{tmpdir}/request_0.json", 'w') as f:
    json.dump(initial_payload, f)

for i in range(1, num_requests + 1):
    payload = {
        "model": os.environ["MODEL"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_questions[i - 1]},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": 1,
    }
    with open(f"{tmpdir}/request_{i}.json", 'w') as f:
        json.dump(payload, f)
PY

send_request() {
  local id=$1
  local start_time=$(date +%s.%N)
  curl -sS -N -X POST "$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$TMPDIR/request_$id.json" | while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
      data="${line#data: }"
      if [[ "$data" != "[DONE]" && -n "$data" ]]; then
        printf '%s' "$data" | python3 -c "
import sys, json
try:
    chunk = json.load(sys.stdin)
    content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
    if content:
        print('[$id] ' + content.replace('\n', '\n[$id] '), end='', flush=True)
except: pass
"
      fi
    fi
  done
  local end_time=$(date +%s.%N)
  local duration=$(echo "$end_time - $start_time" | bc)
  echo "$duration" > "$TMPDIR/time_$id.txt"
  echo "[Request $id] completed in ${duration}s"
}

echo "=== Sending initial request to populate KV cache ==="
send_request 0
echo ""

echo "=== Sending $NUM_REQUESTS follow-up requests with different user messages ==="
for i in $(seq 1 $NUM_REQUESTS); do
  send_request $i &
done

wait

echo ""
echo "=== Timing Summary ==="
python3 - "$TMPDIR" "$NUM_REQUESTS" <<'PY'
import sys, os

tmpdir = sys.argv[1]
num_requests = int(sys.argv[2])

initial_time_file = f"{tmpdir}/time_0.txt"
if os.path.exists(initial_time_file):
    with open(initial_time_file) as f:
        initial_time = float(f.read().strip())
    print(f"Initial request (KV cache population): {initial_time:.3f}s")
    print("")

times = []
for i in range(1, num_requests + 1):
    time_file = f"{tmpdir}/time_{i}.txt"
    if os.path.exists(time_file):
        with open(time_file) as f:
            times.append(float(f.read().strip()))

if times:
    avg = sum(times) / len(times)
    print(f"Follow-up requests completed: {len(times)}/{num_requests}")
    print(f"Min time: {min(times):.3f}s")
    print(f"Max time: {max(times):.3f}s")
    print(f"Average time: {avg:.3f}s")
else:
    print("No timing data collected")
PY
