from experiment import Experiment

experiment = Experiment(
  "extracted_terms-seed_set-token.csv", 
  "seed_set.csv", 
  "population_set_ieee.csv", 
  "extracted_terms_df-seed_set-token.csv", 
  "extracted_terms_df-population_set-token.csv", 
  "occurrence_matrix-seed_set-token.csv", 
  [
    ["deep learning", "neural network"], 
    ["legal", "law"]
  ], 
  [.05, .10, .15, .20, .25, .30], 
  [.001, .01, .02, .05, .10, .20]
)
report = experiment.run()

rep_file_name = f"experiment_01/report-{experiment.timestamp.strftime('%Y-%m-%d_%Hh%Mm%Ss')}.txt"
with open(rep_file_name, 'w') as rep_file:
  rep_file.write(report)

print(f'Done!')