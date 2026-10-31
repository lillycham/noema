{
  description = "noema: small models trained on synthetic worlds, to see what structure they learn";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      # For a rented GPU: the prebuilt CUDA PyTorch, so the host fetches it and compiles nothing.
      rented = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
      rentedPython = rented.python3.withPackages (ps: [ ps.numpy ps.torch-bin ps.matplotlib ]);
      batch = mode: rented.writeShellApplication {
        name = "noema-${mode}";
        runtimeInputs = [ rentedPython rented.gcc rented.findutils rented.coreutils ];
        text = ''
          export PYTHONPATH=${self}
          exec ${rented.bash}/bin/bash ${self}/scripts/rent_rung5.sh ${mode}
        '';
      };
      needs = { minVram = 8; minCuda = 12.8; };
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

      apps.x86_64-linux = {
        smoke = { type = "app"; program = "${batch "smoke"}/bin/noema-smoke"; };
        rung5 = { type = "app"; program = "${batch "batch"}/bin/noema-batch"; };
      };

      # shukka jobs: what each run needs. What to pay for it is the policy of whoever runs it.
      shukkaJobs = {
        smoke = { app = "smoke"; gpu = needs // { models = [ "RTX_3060" "RTX_3070" "RTX_3080" ]; }; disk = 30; maxDuration = "30m"; };
        rung5 = { app = "rung5"; gpu = needs // { minVram = 24; models = [ "RTX_5090" "RTX_4090" ]; }; disk = 40; maxDuration = "90m"; };
      };
    };
}
