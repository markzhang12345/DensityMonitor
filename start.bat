start powershell -NoExit -Command "cd Raspb; python main.py"
start powershell -NoExit -Command "cd Raspb; python data_mock.py"

start powershell -NoExit -Command "cd Server; python server.py"