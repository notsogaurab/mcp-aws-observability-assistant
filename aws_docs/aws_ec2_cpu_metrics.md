
        # AWS EC2 CPU Utilization Metrics

        CPU utilization for EC2 instances is a key metric to monitor. High CPU usage can indicate a bottleneck or an application issue.

        **Key Metrics:**
        * `CPUUtilization`: The percentage of allocated EC2 compute units that are currently in use on the instance.
        * `CPUCreditUsage`: (For T2/T3 instances) The number of CPU credits consumed by the instance.
        * `CPUCreditBalance`: (For T2/T3 instances) The number of CPU credits remaining for the instance.

        **Troubleshooting High CPU Utilization:**
        1.  **Identify the Process:** Log into the EC2 instance and use `top`, `htop`, or Task Manager (Windows) to identify processes consuming high CPU.
        2.  **Application Logs:** Check application logs on the instance for errors or intense operations.
        3.  **Scaling:** Consider scaling up the instance type or scaling out with an Auto Scaling Group.
        4.  **Misconfiguration:** Check for runaway scripts or misconfigured cron jobs.
        