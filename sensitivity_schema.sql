do $$
begin
  create table if not exists "sensitivity" (
    "id"                        serial unique,
    "accuracy"                  double precision, -- the model prediction rate
    "number_of_neurons"         integer,          -- the number of neurons in the reservoire
    "reservoire_connectivity"   double precision, -- the probability of each neurons being connected
    "spectral_radius"           double precision, -- the spectral radius of the connectivity matrix
    "input_connectivity"        double precision, -- the probability of a neuron being connected to the input
    "representation"            text,             -- the representation we choose
    "input_scaling"             double precision, -- the input scaling
    unique (
      number_of_neurons,
      reservoire_connectivity,
      spectral_radius,
      input_connectivity,
      representation,
      input_scaling
      )
    );
end$$;

