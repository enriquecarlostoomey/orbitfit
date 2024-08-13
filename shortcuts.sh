# TODO: This might be overkill, read http://wiki.ros.org/docker/Tutorials/GUI
xhost +local:root

docker_build () {
    docker build -t commissioning .
}

docker_run () {
    docker run --rm -it \
        --volume $(pwd):/home \
        --user=root \
        --env="DISPLAY" \
        --volume="/etc/group:/etc/group:ro" \
        --volume="/etc/passwd:/etc/passwd:ro" \
        --volume="/etc/shadow:/etc/shadow:ro" \
        --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        commissioning \
        $*
    }

alias d=docker_run
alias build=docker_build
