# Dispy

## Links
* Documentation: http://dispy.sourceforge.net/
* Git: https://github.com/pgiri/dispy

## Docker
Create a Dispy container:

```
docker-compose up -d
```

Scale up:

```
docker-compose scale dispy=<number_of_containers>
```

Execute a job (workdir is `/jobs` by default)

```
docker exec dispy_dispy_<x> python3.5 job.py
```
where &lt;x&gt; is a container's instance number

## Parameters
Dispynode parameters can be changed in the `docker-compose.yml` file
Job parameters can be changed directly in the `job.py` file. As this file is included
in the container as a volume, no need to restart the container. You only need to fire
the "Execute Job" command once more to see the result

* `nbjobs` defines the number of jobs to be executed be the nodes
* `n` is the number of lines to train the algorithm with
* `nn` is the number of lines we want to match from the trained algorithm (nn &lt;= n)


## Issues
In some cases, the job may hang indefinitely. In this casz, you need to
re-create all containers, otherwise no further job will be executed

```
docker-compose up -d --force-recreate
```
If you scaled your containers, all of them will be recreated, which is the
intended behaviour.

For instance, running the job on an i7 proc with 16G Ram and 4 executors with 2
CPUs each:

* `nbjobs=4`, `n=1000000` and `nn=1000` works fine
* `nbjobs=4`, `n=1000000` and `nn=10000` hangs indefinitely