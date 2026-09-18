{
  description = "marimo-lab: reproducible marimo research workspace (uv + nix)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default =
          let
            python = pkgs.python313;
            # manylinux wheels (duckdb, pyarrow, scipy, vl-convert, ...) dlopen
            # libstdc++/libz at import time; NixOS has no global /usr/lib.
            runtimeLibs = pkgs.lib.makeLibraryPath (
              [
                pkgs.zlib
                pkgs.openssl
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ pkgs.stdenv.cc.cc.lib ]
            );
          in
          pkgs.mkShell {
            name = "marimo-lab";

            packages = [
              pkgs.uv
              python
              pkgs.git
              pkgs.gnumake
            ];

            env = {
              # Pin uv to the nixpkgs interpreter; never fetch prebuilt CPython,
              # whose dynamic loader path does not exist on NixOS.
              UV_PYTHON = python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
              UV_PROJECT_ENVIRONMENT = ".venv";
              LD_LIBRARY_PATH = runtimeLibs;
              MARIMO_SKIP_UPDATE_CHECK = "1";
            };

            shellHook = ''
              if [ -z "''${MARIMO_LAB_NO_SYNC:-}" ]; then
                uv sync --quiet || echo "marimo-lab: 'uv sync' failed; run it manually" >&2
              fi
              export VIRTUAL_ENV="$PWD/.venv"
              export PATH="$VIRTUAL_ENV/bin:$PATH"

              echo "marimo-lab ready  ·  python $(${python.interpreter} -V | cut -d' ' -f2)  ·  uv $(uv --version | cut -d' ' -f2)"
              echo "  make edit          open the notebook workspace"
              echo "  make new N=name    create notebooks/name.py"
              echo "  make check         ruff + pytest"
            '';
          };
      });
    };
}
