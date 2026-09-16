"""Personal T1D twins fitted to InSite data.

Pipeline: raw day records -> ``data.build_timeline`` -> ``fit.fit_twin`` (a
posterior over this person's physiology and context sensitivities) ->
``experiment.run_settings_experiment`` (paired CR/ISF/basal arms) and
``population`` (distributions for synthetic people). ``synthetic`` generates
people with known parameters to check recovery.
"""
