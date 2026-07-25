rule clone_namaster:
    output: directory("resources/NaMaster")
    shell: "git clone https://github.com/LSSTDESC/NaMaster.git resources/NaMaster"

rule install_namaster:
    input: directory("resources/NaMaster")
    shell:
        """
        cd resources/NaMaster
        singularity exec -w -e instance://sptlab python setup.py install --user
        """