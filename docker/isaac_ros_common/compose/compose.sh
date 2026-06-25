#!/bin/bash
#!/bin/bash

PLATFORM="$(uname -m)"

echo "Platform $PLATFORM detected"

if [[ $PLATFORM == "aarch64" ]]; then
    cat docker-compose-aarch64.txt > docker-compose.yml
else
    cat docker-compose-x86_64.txt > docker-compose.yml
fi

docker compose up -d

docker exec -it aerostack2_container /bin/bash