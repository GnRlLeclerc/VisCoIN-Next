{
  description = "VisCoIN model implementation using PyTorch";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      # Supported systems
      systems = [
        "aarch64-linux"
        "i686-linux"
        "x86_64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      # Add the lpips package + override torch to use the bin version
      overlays = [
        (final: prev: {
          python312 = prev.python312.override {
            packageOverrides = finalPy: prevPy: {
              lpips = final.python312.pkgs.buildPythonPackage rec {
                pname = "lpips";
                version = "0.1.4";

                src = final.pkgs.fetchFromGitHub {
                  owner = "richzhang";
                  repo = "PerceptualSimilarity";
                  rev = version;
                  sha256 = "sha256-dIQ9B/HV/2kUnXLXNxAZKHmv/Xv37kl2n6+8IfwIALE=";
                };

                dependencies = with final.python312.pkgs; [
                  torch
                  torchvision
                  numpy
                  opencv4
                  scikit-image
                  matplotlib
                  tqdm
                  ipykernel
                ];
              };

              # Temporary fix for python312Packages.triton-bin
              # Follow this issue https://github.com/NixOS/nixpkgs/issues/351717
              triton-bin = prevPy.triton-bin.overridePythonAttrs (oldAttrs: {
                postFixup = ''
                  chmod +x "$out/${prev.python312.sitePackages}/triton/backends/nvidia/bin/ptxas"
                  substituteInPlace $out/${prev.python312.sitePackages}/triton/backends/nvidia/driver.py \
                    --replace \
                      'return [libdevice_dir, *libcuda_dirs()]' \
                      'return [libdevice_dir, "${prev.addDriverRunpath.driverLink}/lib", "${prev.cudaPackages.cuda_cudart}/lib/stubs/"]'
                '';
              });

              # Replace main torch packages with their bin versions for CUDA support + caching
              torch = finalPy.torch-bin;
              torchvision = finalPy.torchvision-bin;
              torchaudio = finalPy.torchaudio-bin;
            };
          };

          python312Packages = final.python312.pkgs;
        })

      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShell = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system overlays;
            config.allowUnfree = true;
          };
        in
        pkgs.mkShell {
          buildInputs = with pkgs; [
            # For Numpy, Torch, etc.
            stdenv.cc.cc
            zlib

            # Plotting with GTK backend
            gtk3
            gobject-introspection

            # GTK SVG image support
            librsvg
          ];

          packages = with pkgs; [
            # Build system for loading C++ extensions in torch
            ninja
            cudatoolkit

            (python312.withPackages (
              ps: with ps; [
                # Deep learning libraries
                torch-bin
                torchvision-bin
                transformers
                pillow
                tqdm
                numpy
                lpips
                clip
                ipykernel

                # GTK backend for matplotlib
                pygobject3

                # Utilities
                click
                matplotlib
              ]
            ))
          ];

          MPLBACKEND = "GTK3Agg";
          CUDA_HOME = pkgs.cudatoolkit;
          TORCH_CUDA_ARCH_LIST = "8.6"; # Compute capability of RTX 3050
        }
      );
    };
}
