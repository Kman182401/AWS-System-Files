
import psutil
import platform
import os
import subprocess
import json
from datetime import datetime
import time

class PCDiagnosticBot:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        
    def add_issue(self, category, message, severity="HIGH"):
        self.issues.append({
            "category": category,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_warning(self, category, message):
        self.warnings.append({
            "category": category,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_info(self, category, message):
        self.info.append({
            "category": category,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def check_cpu_usage(self):
        """Check CPU usage and temperature"""
        print("🔍 Checking CPU performance...")
        
        # Get CPU usage over 5 seconds
        cpu_percent = psutil.cpu_percent(interval=5)
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        if cpu_percent > 80:
            self.add_issue("CPU", f"High CPU usage detected: {cpu_percent}%")
        elif cpu_percent > 60:
            self.add_warning("CPU", f"Elevated CPU usage: {cpu_percent}%")
        else:
            self.add_info("CPU", f"CPU usage normal: {cpu_percent}%")
        
        # Check CPU frequency
        if cpu_freq:
            if cpu_freq.current < cpu_freq.max * 0.5:
                self.add_warning("CPU", f"CPU running at low frequency: {cpu_freq.current:.0f}MHz (Max: {cpu_freq.max:.0f}MHz)")
        
        self.add_info("CPU", f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
        
        # Try to get CPU temperature (Windows)
        try:
            if platform.system() == "Windows":
                result = subprocess.run(['wmic', 'cpu', 'get', 'temperature', '/format:list'], 
                                      capture_output=True, text=True, timeout=10)
                # Note: Most consumer CPUs don't report temperature via WMI
        except:
            pass
    
    def check_memory_usage(self):
        """Check RAM usage and swap"""
        print("🔍 Checking memory usage...")
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        if memory.percent > 85:
            self.add_issue("MEMORY", f"High RAM usage: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
        elif memory.percent > 70:
            self.add_warning("MEMORY", f"Elevated RAM usage: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
        else:
            self.add_info("MEMORY", f"RAM usage normal: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
        
        if swap.percent > 50:
            self.add_issue("MEMORY", f"High swap usage: {swap.percent}% - Consider adding more RAM")
        elif swap.percent > 20:
            self.add_warning("MEMORY", f"Moderate swap usage: {swap.percent}%")
    
    def check_disk_usage(self):
        """Check disk space and health"""
        print("🔍 Checking disk usage...")
        
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                percent = (partition_usage.used / partition_usage.total) * 100
                
                if percent > 90:
                    self.add_issue("DISK", f"Disk {partition.device} critically full: {percent:.1f}%")
                elif percent > 80:
                    self.add_warning("DISK", f"Disk {partition.device} getting full: {percent:.1f}%")
                else:
                    self.add_info("DISK", f"Disk {partition.device} usage: {percent:.1f}%")
            except PermissionError:
                continue
        
        # Check disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.add_info("DISK", f"Disk reads: {disk_io.read_count}, Disk writes: {disk_io.write_count}")
    
    def check_running_processes(self):
        """Check for resource-heavy processes"""
        print("🔍 Checking running processes...")
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 0 or proc_info['memory_percent'] > 0:
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage
        cpu_heavy = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:10]
        memory_heavy = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:10]
        
        for proc in cpu_heavy[:5]:
            if proc['cpu_percent'] and proc['cpu_percent'] > 10:
                self.add_warning("PROCESSES", f"High CPU process: {proc['name']} ({proc['cpu_percent']:.1f}% CPU)")
        
        for proc in memory_heavy[:5]:
            if proc['memory_percent'] and proc['memory_percent'] > 5:
                self.add_warning("PROCESSES", f"High memory process: {proc['name']} ({proc['memory_percent']:.1f}% RAM)")
        
        self.add_info("PROCESSES", f"Total running processes: {len(processes)}")
    
    def check_network_usage(self):
        """Check network statistics"""
        print("🔍 Checking network usage...")
        
        net_io = psutil.net_io_counters()
        if net_io:
            self.add_info("NETWORK", f"Bytes sent: {net_io.bytes_sent // (1024**2):.1f}MB, Bytes received: {net_io.bytes_recv // (1024**2):.1f}MB")
            self.add_info("NETWORK", f"Packets sent: {net_io.packets_sent}, Packets received: {net_io.packets_recv}")
            
            if net_io.errin > 100 or net_io.errout > 100:
                self.add_warning("NETWORK", f"Network errors detected - In: {net_io.errin}, Out: {net_io.errout}")
    
    def check_system_temperatures(self):
        """Check system temperatures (if available)"""
        print("🔍 Checking system temperatures...")
        
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > 80:
                            self.add_issue("TEMPERATURE", f"{name} - {entry.label or 'Unknown'}: {entry.current}°C (HIGH!)")
                        elif entry.current > 70:
                            self.add_warning("TEMPERATURE", f"{name} - {entry.label or 'Unknown'}: {entry.current}°C")
                        else:
                            self.add_info("TEMPERATURE", f"{name} - {entry.label or 'Unknown'}: {entry.current}°C")
            else:
                self.add_info("TEMPERATURE", "Temperature sensors not available or not supported")
        except AttributeError:
            self.add_info("TEMPERATURE", "Temperature monitoring not supported on this system")
    
    def check_battery_health(self):
        """Check battery status (for laptops)"""
        print("🔍 Checking battery status...")
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                if battery.percent < 20 and not battery.power_plugged:
                    self.add_warning("BATTERY", f"Low battery: {battery.percent}%")
                
                self.add_info("BATTERY", f"Battery: {battery.percent}%, Plugged in: {battery.power_plugged}")
                
                if hasattr(battery, 'secsleft') and battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                    hours = battery.secsleft // 3600
                    minutes = (battery.secsleft % 3600) // 60
                    self.add_info("BATTERY", f"Time remaining: {hours}h {minutes}m")
            else:
                self.add_info("BATTERY", "No battery detected (desktop system)")
        except AttributeError:
            self.add_info("BATTERY", "Battery monitoring not supported")
    
    def check_startup_programs(self):
        """Check startup programs (Windows)"""
        print("🔍 Checking startup programs...")
        
        if platform.system() == "Windows":
            try:
                # This is a simplified check - in a real implementation you'd want to check registry entries
                startup_count = len([proc for proc in psutil.process_iter(['name']) if proc.info['name']])
                if startup_count > 50:
                    self.add_warning("STARTUP", f"Many programs running: {startup_count}. Consider disabling unnecessary startup programs")
                else:
                    self.add_info("STARTUP", f"Programs running: {startup_count}")
            except:
                pass
    
    def get_system_info(self):
        """Get basic system information"""
        print("🔍 Gathering system information...")
        
        uname = platform.uname()
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        
        self.add_info("SYSTEM", f"System: {uname.system} {uname.release}")
        self.add_info("SYSTEM", f"Machine: {uname.machine}")
        self.add_info("SYSTEM", f"Processor: {uname.processor}")
        self.add_info("SYSTEM", f"Boot time: {boot_time}")
        self.add_info("SYSTEM", f"Uptime: {str(uptime).split('.')[0]}")
    
    def run_full_diagnostic(self):
        """Run all diagnostic checks"""
        print("🚀 Starting PC Diagnostic Scan...")
        print("=" * 50)
        
        self.get_system_info()
        self.check_cpu_usage()
        self.check_memory_usage()
        self.check_disk_usage()
        self.check_running_processes()
        self.check_network_usage()
        self.check_system_temperatures()
        self.check_battery_health()
        self.check_startup_programs()
        
        print("\n" + "=" * 50)
        print("📊 DIAGNOSTIC RESULTS")
        print("=" * 50)
        
        # Display critical issues
        if self.issues:
            print("\n🚨 CRITICAL ISSUES FOUND:")
            for issue in self.issues:
                print(f"  ❌ [{issue['category']}] {issue['message']}")
        
        # Display warnings
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠️  [{warning['category']}] {warning['message']}")
        
        # Display info
        if self.info:
            print("\n✅ SYSTEM INFORMATION:")
            for info in self.info:
                print(f"  ℹ️  [{info['category']}] {info['message']}")
        
        # Summary
        print(f"\n📈 SUMMARY:")
        print(f"  • Critical Issues: {len(self.issues)}")
        print(f"  • Warnings: {len(self.warnings)}")
        print(f"  • Info Items: {len(self.info)}")
        
        if not self.issues and not self.warnings:
            print("\n🎉 Great! No major issues detected. Your system appears to be running well!")
        elif self.issues:
            print(f"\n🔧 RECOMMENDATIONS:")
            print("  • Address critical issues immediately")
            print("  • Consider restarting your computer if issues persist")
            print("  • Check Task Manager for resource-heavy programs")
            print("  • Ensure adequate cooling and ventilation")
        
        return {
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main function to run the diagnostic bot"""
    try:
        bot = PCDiagnosticBot()
        results = bot.run_full_diagnostic()
        
        # Save results to file
        with open(f"pc_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: pc_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Diagnostic cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during diagnostic: {str(e)}")

if __name__ == "__main__":
    main()
