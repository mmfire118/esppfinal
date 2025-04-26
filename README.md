# real 9 % discount, base run
python geothermal_model.py \
  --data_dir ./prices --capacities 30 40 \
  --line_miles 5 10 --tax_credits itc30 ptc \
  --price_floor -10 --discount 0.09 --basis_adj 3.0

# nominal 9 % WACC, with sensitivity sweep
python geothermal_model.py \
  --data_dir ./prices --disc_mode nom --sens
