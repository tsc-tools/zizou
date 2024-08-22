# Some notes on how to setup a development environment for zizou using VS Code and devcontainers

The biggest headache is to get the permissions right. The only way I could make it work was
by setting the following two variables in my `.bash_aliases` or `.bashrc` file:
```
export D_UID=`id -u`
export D_GID=`id -g`
```

After that do the following in VS Code:
```
Ctrl-Shift P
dev containers: Rebuild and Reopen in Container
```

This will build the devcontainer matching the container user's uid and gid with your local host one. 