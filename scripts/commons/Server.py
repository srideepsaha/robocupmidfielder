import subprocess

class Server():
    def __init__(self, first_server_p, first_monitor_p, n_servers) -> None:
        try:
            import psutil
            self.check_running_servers(psutil, first_server_p, first_monitor_p, n_servers)
        except ModuleNotFoundError:
            print("Info: Cannot check if the server is already running, because the psutil module was not found")
            
        self.first_server_p = first_server_p
        self.n_servers = n_servers
        self.rcss_processes = []

        # makes it easier to kill test servers without affecting train servers
        cmd = "simspark" if n_servers == 1 else "rcssserver3d"
        for i in range(n_servers):
            full_cmd = f"{cmd} --agent-port {first_server_p+i} --server-port {first_monitor_p+i}"
            print(f"[Server] Launching server #{i} with: {full_cmd}")
            self.rcss_processes.append(
                subprocess.Popen((f"{cmd} --agent-port {first_server_p+i} --server-port {first_monitor_p+i}").split(),
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
            )

    def check_running_servers(self, psutil, first_server_p, first_monitor_p, n_servers):
        ''' Check if any server is running on chosen ports '''
        found = False

        
        p_list = []
        for p in psutil.process_iter():
            try:
                if p.name() in ["rcssserver3d", "simspark"] and p.cmdline():
                    p_list.append(p)
            except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
                continue

        range1 = (first_server_p, first_server_p + n_servers)
        range2 = (first_monitor_p, first_monitor_p + n_servers)
        bad_processes = []

        for p in p_list:
            try:
                ports = [int(arg) for arg in p.cmdline()[1:] if arg.isdigit()]
            except Exception:
                ports = []
            if len(ports) == 0:
                ports = [3100, 3200]  # fallback to default

            conflicts = [str(port) for port in ports if (
                (range1[0] <= port < range1[1]) or (range2[0] <= port < range2[1]))]

            if conflicts:
                if not found:
                    print("\nThere are already servers running on the same port(s)!")
                    found = True
                bad_processes.append(p)
                print(f"Port(s) {','.join(conflicts)} already in use by \"{' '.join(p.cmdline())}\" (PID:{p.pid})")

        if found:
            print()
            while True:
                inp = input("Enter 'kill' to kill these processes or ctrl+c to abort. ")
                if inp == "kill":
                    for p in bad_processes:
                        p.kill()
                    return

            

    def kill(self):
        for p in self.rcss_processes:
            p.kill()
        print(f"Killed {self.n_servers} rcssserver3d processes starting at {self.first_server_p}")
