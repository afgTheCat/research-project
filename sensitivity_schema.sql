do $$
begin
  create table if not exists "output_sensitivity_test" (
    "id"                        serial unique,
    "accuracy"                  double precision not null, -- the model prediction rate
    "number_of_neurons"         integer          not null, -- the number of neurons in the reservoire
    "reservoire_connectivity"   double precision not null, -- the probability of each neurons being connected
    "spectral_radius"           double precision,          -- the spectral radius of the connectivity matrix
    "input_connectivity"        double precision not null, -- the probability of a neuron being connected to the input
    "representation"            text not null,             -- the representation we choose
    "input_scaling"             double precision not null, -- the input scaling
    "thalmic_mean"              double precision not null, -- thalmic mean
    "connectivity_strength"     double precision not null  -- avg connection strength between the neurons
    );

  create table if not exists "representation_test" (
    "id"                        serial unique,
    "accuracy"                  double precision not null, -- the model prediction rate
    "number_of_neurons"         integer          not null, -- the number of neurons in the reservoire
    "reservoire_connectivity"   double precision not null, -- the probability of each neurons being connected
    "spectral_radius"           double precision,          -- the spectral radius of the connectivity matrix
    "input_connectivity"        double precision not null, -- the probability of a neuron being connected to the input
    "representation"            text not null,             -- the representation we choose
    "input_scaling"             double precision not null, -- the input scaling
    "thalmic_mean"              double precision not null, -- thalmic mean
    "connectivity_strength"     double precision not null  -- avg connection strength between the neurons
    );
end$$;

