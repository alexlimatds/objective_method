from experiment import Experiment

experiment = Experiment(
  "extracted_terms-seed_set-pos_tag.csv", 
  "seed_set.csv", 
  "population_set_ieee.csv", 
  "extracted_terms_df-seed_set-pos_tag.csv", 
  "extracted_terms_df-population_set-pos_tag.csv", 
  "occurrence_matrix-seed_set-pos_tag.csv", 
  [
    ["deep learning", "neural network"], 
    ["legal", "law"]
  ], 
  [.05, .10, .15, .20, .25, .30], 
  [.001, .01, .02, .05, .10, .20]
)
report = experiment.run()

rep_file_name = f"experiment_02/report-{experiment.timestamp.strftime('%Y-%m-%d_%Hh%Mm%Ss')}.txt"
with open(rep_file_name, 'w') as rep_file:
  rep_file.write(report)

print(f'Done!')