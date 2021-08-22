from gremlin_python.process.traversal import Column
from typing import List
import time
import logging
from gremlin_python.driver.client import Client
from datetime import datetime
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import\
    DriverRemoteConnection
from gremlin_python.process.traversal import T
from gremlin_python.process.traversal import P
from gremlin_python.process.graph_traversal import __
import json
import uuid
from gremlin_python.process.traversal import Order
import nest_asyncio
from write_cot import parse_cftc_to_json
nest_asyncio.apply()


server = '78.47.115.9'
port = 8182
endpoint = 'ws://' + server + ':' + str(port) + '/gremlin'
transport_args = {'max_content_length': 200000}

connection = DriverRemoteConnection(endpoint, 'g', **transport_args)
# Connection 1 to read data
g = traversal().withRemote(connection)
# Connection 2 to write data
client = Client(endpoint, 'g')
