start powershell -NoExit -Command "cd Raspb; python http_client.py"
start powershell -NoExit -Command "cd Raspb; python data_mock.py"

start powershell -NoExit -Command "cd Server; python server.py"