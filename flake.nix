{
  description = "noema: small models trained on synthetic worlds, to see what structure they learn";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      # A stock PyTorch image for rented GPUs: the host compiles nothing. Nix would build all of CUDA there,
      # as it is unfree and so not in the public cache. The code comes from GitHub, so push the branch first.
      rent = mode: disk: maxDuration: gpu: {
        image = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime";
        entrypoint = [ "bash" "-c" ''
          set -eu
          export PATH=/opt/conda/bin:$PATH
          python -c "import urllib.request as u; u.urlretrieve('https://codeload.github.com/lillycham/noema/tar.gz/refs/heads/feat/rung5', '/tmp/noema.tgz')"
          mkdir -p /tmp/noema && tar xzf /tmp/noema.tgz --strip-components=1 -C /tmp/noema
          python -m pip install -q numpy matplotlib
          DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc
          PYTHONPATH=/tmp/noema exec bash /tmp/noema/scripts/rent_rung5.sh ${mode}
        '' ];
        inherit gpu disk maxDuration;
      };
    in
    {
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: [ ps.numpy ps.torch ps.matplotlib ps.pytest ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            # A fixed path to this Python for editors, because the store path changes on every update.
            shellHook = ''
              ln -sfn ${pythonEnv} .python-env
            '';
          };
        });

      # shukka jobs: what each run needs. What to pay for it is the policy of whoever runs it.
      shukkaJobs = {
        smoke = rent "smoke" 30 "20m" { minVram = 8; minCuda = 12.9; models = [ "RTX_3060" "RTX_3070" "RTX_3080" ]; };
        rung5 = rent "batch" 40 "90m" { minVram = 24; minCuda = 12.9; models = [ "RTX_5090" "RTX_4090" ]; };
      };
    };
}
