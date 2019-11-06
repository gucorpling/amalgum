pkill -f parsoid
python main.py \
  --config configs/wikinews.yaml \
  --method scrape \
  --output-dir ../out/news/ \
  --stop-after 170 \
  --cmtitle=Category:Africa \
  --cmtitle=Category:Middle_East \
  --cmtitle=Category:Asia \
  --cmtitle=Category:Europe \
  --cmtitle=Category:North_America \
  --cmtitle=Category:South_America \
