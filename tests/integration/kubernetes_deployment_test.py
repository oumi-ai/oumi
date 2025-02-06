import os
import subprocess
import time
import unittest

class TestKubernetesDeployment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.helm_chart_path = "../../kubernetes/helm"
        cls.timoni_module_path = "../../kubernetes/timoni"
        cls.values_file = "../../kubernetes/helm/values.yaml"

    def test_helm_deployment(self):
        # Install Helm chart
        subprocess.run(["helm", "install", "oumi", self.helm_chart_path, "-f", self.values_file], check=True)
        time.sleep(10)  # Wait for the deployment to stabilize

        # Check if the deployment is running
        result = subprocess.run(["kubectl", "get", "deployments"], capture_output=True, text=True)
        self.assertIn("oumi", result.stdout)

        # Uninstall Helm chart
        subprocess.run(["helm", "uninstall", "oumi"], check=True)

    def test_timoni_deployment(self):
        # Apply Timoni module
        subprocess.run(["timoni", "apply", "-f", os.path.join(self.timoni_module_path, "module.yaml"), "-f", self.values_file], check=True)
        time.sleep(10)  # Wait for the deployment to stabilize

        # Check if the deployment is running
        result = subprocess.run(["kubectl", "get", "deployments"], capture_output=True, text=True)
        self.assertIn("oumi", result.stdout)

        # Delete Timoni module
        subprocess.run(["timoni", "delete", "-f", os.path.join(self.timoni_module_path, "module.yaml")], check=True)

if __name__ == "__main__":
    unittest.main()
