import resframe
import scipy.io
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool
import psycopg2


def create_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="user",
            password="password",
            database="db",
        )
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def if_none_null(val):
    return "null" if val is None else val


def build_query(
    table_name,
    accuracy,
    number_of_neurons,
    reservoire_connectivity,
    spectral_radius,
    input_connectivity,
    representation,
    input_scaling,
    thalmic_mean,
):
    return (
        f"insert into {table_name} ("
        f"  accuracy,"
        f"  number_of_neurons,"
        f"  reservoire_connectivity,"
        f"  spectral_radius,"
        f"  input_connectivity,"
        f"  representation,"
        f"  input_scaling,"
        f"  thalmic_mean"
        f") values("
        f"  {accuracy},"
        f"  {number_of_neurons},"
        f"  {reservoire_connectivity},"
        f"  {if_none_null(spectral_radius)},"
        f"  {input_connectivity},"
        f"  '{representation}',"
        f"  {input_scaling},"
        f"  {thalmic_mean}"
        f");"
    )


def insert_to_db(
    table_name,
    accuracy,
    number_of_neurons,
    reservoire_connectivity,
    spectral_radius,
    input_connectivity,
    representation,
    input_scaling,
    thalmic_mean,
):
    q = build_query(
        table_name,
        accuracy,
        number_of_neurons,
        reservoire_connectivity,
        spectral_radius,
        input_connectivity,
        representation,
        input_scaling,
        thalmic_mean,
    )
    connection = create_connection()
    if connection is not None:
        cur = connection.cursor()
        cur.execute(q)
        cur.close()
        connection.commit()
    else:
        raise RuntimeError("Connection to db could not be established")


def search_for_param(input_params):
    table_name = input_params["table_name"]

    Xtrain = input_params["data"]["Xtrain"]
    Xtest = input_params["data"]["Xtest"]
    Ytrain = input_params["data"]["Ytrain"]
    Ytest = input_params["data"]["Ytest"]

    number_of_neurons = 100

    rc_model = resframe.RCModel(
        dt=1,
        number_of_neurons=number_of_neurons,
        **input_params["reservoire_parameters"],
    )
    rc_model.train(Xtrain, Ytrain)
    accuracy, f1 = rc_model.test(Xtest, Ytest)

    erdos_connectivity = input_params["reservoire_parameters"].get("erdos_connectivity")
    uniform_lower = input_params["reservoire_parameters"].get("erdos_uniform_lower")
    representation = input_params["reservoire_parameters"].get("representation")
    input_scale = input_params["reservoire_parameters"].get("input_scale")
    spectral_radius = input_params["reservoire_parameters"].get("spectral_radius")
    input_connectivity = input_params["reservoire_parameters"].get(
        "input_connectivity_p"
    )
    thalmic_mean = input_params["reservoire_parameters"].get("thalmic_mean")

    insert_to_db(
        table_name,
        accuracy,
        number_of_neurons,
        erdos_connectivity,
        spectral_radius,
        input_connectivity,
        representation,
        input_scale,
        thalmic_mean,
    )
    print(
        (
            f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}, erdos conn: {erdos_connectivity:.3f},",
            f"uniform lower: {uniform_lower:.3f} input connectivity: {input_connectivity:.3f},",
            f"representation: {representation}, input scale: {input_scale:.3f}",
        )
    )


def sensitivity_analysis(reservoire_params, table_name):
    data = scipy.io.loadmat("./data/JpVow.mat")
    onehot_encoder = OneHotEncoder(sparse=False)

    Xtrain = data["X"]
    Ytrain = data["Y"]
    Xtest = data["Xte"]
    Ytest = data["Yte"]

    Ytrain = onehot_encoder.fit_transform(Ytrain)
    Ytest = onehot_encoder.transform(Ytest)

    data = {
        "Xtrain": Xtrain,
        "Ytrain": Ytrain,
        "Xtest": Xtest,
        "Ytest": Ytest,
    }

    params = [
        {
            "data": data,
            "reservoire_parameters": reservoire_param,
            "table_name": table_name,
        }
        for reservoire_param in reservoire_params
    ]

    with Pool() as p:
        p.map(search_for_param, params)
