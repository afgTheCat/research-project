use crate::izikevich_model::InputStep;

pub fn test_input() -> Vec<InputStep> {
    let input_vals = [
        [
            7.617594143940083,
            5.161431364958054,
            -1.442556045036287,
            -18.062510947421824,
            -9.566983178467233,
            -10.404767016241987,
            9.48894198856054,
            25.825663339823215,
            0.2521123622538184,
            -0.32174191813770486,
            7.623731673462246,
            -1.910133935319651,
        ],
        [
            7.343929368546004,
            6.167060833844598,
            -2.0038790152616768,
            -21.225726333863584,
            -9.041290117162147,
            -11.007726624373683,
            9.027714455946136,
            26.250801583802893,
            1.5704881208593813,
            1.3253883953237873,
            5.773318408824927,
            -0.7240404364788565,
        ],
        [
            7.366116187591142,
            5.775856324241655,
            3.6169267029284895,
            -15.919076090201408,
            -5.329483921789442,
            -5.635866854183536,
            12.243097359795323,
            25.34987800733747,
            0.39677576563022043,
            -0.2795801712770325,
            4.931011191953937,
            3.1006995314618893,
        ],
        [
            4.219299356607975,
            0.26875503659169786,
            3.3775844099945234,
            -12.576451238218114,
            2.446151091432953,
            2.738552344065311,
            14.328016976768707,
            25.22803831328985,
            1.9801369556781998,
            -0.6084931092610888,
            -5.369202774354179,
            3.817607994574848,
        ],
        [
            5.593528782455442,
            1.755664460480775,
            1.4009752697099367,
            -13.737955112782522,
            3.0807504548786415,
            0.4313362260443837,
            14.365723893513302,
            23.0267480849715,
            -2.150972604658238,
            -0.5735151083158272,
            -1.1456849338443416,
            6.627544238623058,
        ],
        [
            6.0259872054331955,
            5.4577294816073385,
            1.4862054489929601,
            -9.367182301594532,
            3.9342028103434803,
            -2.3982785356165115,
            11.344953332993953,
            15.988183101601688,
            -7.755081583394677,
            0.9662866203380771,
            12.815109150038674,
            11.089544884163065,
        ],
        [
            7.293055525615336,
            4.676138400611667,
            2.9794780637996277,
            -9.1741103917833,
            6.164050986413661,
            0.43395029863851,
            7.194525719016033,
            16.744067007418696,
            -8.704608274113056,
            -2.7540385559963902,
            14.496139297751418,
            11.015463036416566,
        ],
        [
            3.3116287974277823,
            6.2679995210985675,
            2.179577047861913,
            -6.51741868679485,
            8.730538909121112,
            -2.960185321872257,
            11.072756527743904,
            14.82754720126656,
            -11.096241609314536,
            0.9960478534161965,
            19.691101060357084,
            10.049655243573298,
        ],
        [
            2.0021143446916416,
            5.914931380546759,
            1.9704391342528078,
            -4.973497113772439,
            10.969341851078662,
            -5.972072236232732,
            8.687483953872366,
            10.922731002316366,
            -9.840663403205577,
            7.868840195629039,
            19.441327528205314,
            6.853787849836179,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    input_vals
        .iter()
        .map(|e| InputStep::new(25.0, e.to_vec()))
        .collect()
}
