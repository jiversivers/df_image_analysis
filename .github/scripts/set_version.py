import toml
import subprocess

def set_version():
    # Get the current Git tag
    version = subprocess.check_output(["git", "describe", "--tags"]).strip().decode("utf-8")

    # Load the pyproject.toml file
    toml_file = "pyproject.toml"
    with open(toml_file, "r") as f:
        config = toml.load(f)

    # Set the version in pyproject.toml
    config["project"]["version"] = version
    # Write the updated version back to the pyproject.toml file
    with open(toml_file, "w") as f:
        toml.dump(config, f)

if __name__ == "__main__":
    set_version()
