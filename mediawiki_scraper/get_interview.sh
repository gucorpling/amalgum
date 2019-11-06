pkill -f parsoid
python main.py \
  --config configs/wikinews_interview.yaml \
  --method scrape \
  --output-dir ../out/interview/ \
  --stop-after 960 \
  --split-if-long

