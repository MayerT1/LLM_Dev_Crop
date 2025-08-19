import json
import psycopg2
from sshtunnel import SSHTunnelForwarder
from datetime import datetime, timedelta
from .dssat_plot import get_stress_series_data, get_anomaly_series_data, get_columnRange_series_data
from .dssat_base import Session, AdminBase
import traceback

f = open('./config.json', )
config = json.load(f)


def connect(dbname):
    con = psycopg2.connect(
        database=config['USERNAME'],
        user=config['DBUSER'],
        password=config['PASSWORD'],
        host=config['HOST'],
        port=5432,
    )
    return con


def get_session(con_params, session_data):
    """
    Helper function to recreate the Session object from session data.
    """
    if not con_params or not session_data:
        return None

    # Re-establish the database connection
    con = psycopg2.connect(**con_params)
    # Recreate the Session object
    return Session(AdminBase(con, session_data['admin1_country'], session_data['admin1_name']))


def create_ssh_tunnel():
    """Establish an SSH tunnel and return the tunnel object."""
    ec2_host = config['EC2_HOST']
    ec2_user = config['EC2_USER']
    private_key_path = config['PPK_PATH']
    rds_host = config['RDS_HOST']
    rds_port = int(config['RDS_PORT'])
    local_port = int(config['LOCAL_PORT'])
    tunnel = SSHTunnelForwarder(
        (ec2_host, 22),
        ssh_username=ec2_user,
        ssh_pkey=private_key_path,
        remote_bind_address=(rds_host, rds_port),
        local_bind_address=('127.0.0.1', local_port)
    )
    tunnel.start()  # Important: start the tunnel
    print(f"Tunnel started on local port {tunnel.local_bind_port}")
    return tunnel


def close_ssh_tunnel(tunnel):
    """Close the SSH tunnel."""
    if tunnel.is_active:
        tunnel.stop()
        print("Tunnel closed.")


def run_experiment(planting_date, fert_plan, cultivar, admin1_country, admin1_name):
    tunnel = create_ssh_tunnel()

    con_params = {
        "database": config['USERNAME'],
        "user": config['DBUSER'],
        "password": config['PASSWORD'],
        "host": config['HOST'],
        "port": config['LOCAL_PORT']
    }
    session_data = {
        "admin1_country": admin1_country,
        "admin1_name": admin1_name,
    }

    try:
        # Retrieve and recreate the session object
        session = get_session(con_params, session_data)
        print('DSSAT session established')

        # Update session parameters
        session.simPars.planting_date = datetime.strptime(planting_date, '%Y-%m-%d')
        session.simPars.cultivar = cultivar
        if fert_plan:
            session.simPars.nitrogen_rate = [int(app[0]) for app in fert_plan]
            session.simPars.nitrogen_dap = [int(app[1]) for app in fert_plan]
        else:

            session.simPars.nitrogen_rate = []
            session.simPars.nitrogen_dap = []
        session.run_experiment(fakerun=False)

        ############

        # Update charts with new data
        new_chart_data_range = get_columnRange_series_data(session)

        new_chart_data_water = get_stress_series_data(session, stresstype="water")

        new_chart_data_nitro = get_stress_series_data(session, stresstype="nitrogen")

        return (new_chart_data_range, new_chart_data_nitro, new_chart_data_water)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        close_ssh_tunnel(tunnel)