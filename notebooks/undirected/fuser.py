import pickle

samples = []
with open("cache/base_exp3_figure_polbooks_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/interval_ends_exp3_figure_polbooks_samples_data.pkl", "rb") as f:
    interval_ends = pickle.load(f)

samples = [{**dict1, **dict2} for dict1, dict2 in zip(samples, interval_ends)]

print(len(samples))
with open("cache/exp3_figure_polbooks_samples_data.pkl", "wb") as f:
    pickle.dump(samples, f)

###

samples = []
with open("cache/44_exp3_figure_ca_GRQC_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/55_exp3_figure_ca_GRQC_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/60_exp3_figure_ca_GRQC_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/interval_ends_exp3_figure_ca_GRQC_samples_data.pkl", "rb") as f:
    interval_ends = pickle.load(f)

samples = [{**dict1, **dict2} for dict1, dict2 in zip(samples, interval_ends)]

print(len(samples))
with open("cache/exp3_figure_ca_GRQC_samples_data.pkl", "wb") as f:
    pickle.dump(samples, f)

##################

samples = []
with open("cache/42_exp3_figure_enron_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/55_exp3_figure_enron_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/77_exp3_figure_enron_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/interval_ends_exp3_figure_enron_samples_data.pkl", "rb") as f:
    interval_ends = pickle.load(f)

samples = [{**dict1, **dict2} for dict1, dict2 in zip(samples, interval_ends)]

print(len(samples))
with open("cache/exp3_figure_enron_samples_data.pkl", "wb") as f:
    pickle.dump(samples, f)


##################

samples = []
with open("cache/42_exp3_comp_enron_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

with open("cache/45_exp3_comp_enron_samples_data.pkl", "rb") as f:
    samples.extend(pickle.load(f))

print(len(samples))
with open("cache/exp3_comp_enron_samples_data.pkl", "wb") as f:
    pickle.dump(samples, f)


##################

samples = []
with open("cache/cc_closeness_synth.pkl", "rb") as f:
    cc_closeness_synth = pickle.load(f)

with open("cache/kanon_closeness_synth.pkl", "rb") as f:
    kanon_closeness_synth = pickle.load(f)


samples = [{**dict1, **dict2} for dict1, dict2 in zip(cc_closeness_synth, kanon_closeness_synth)]

print(len(samples))
with open("cache/closeness_synth.pkl", "wb") as f:
    pickle.dump(samples, f)
