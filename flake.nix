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

      # Add the lpips package
      overlays = [
        (final: prev: {
          python312 = prev.python312.override {
            packageOverrides = finalPy: prevPy: {
              lpips = final.python312.pkgs.buildPythonPackage rec {
                pname = "lpips";
                version = "v0.1.4";

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
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
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
                torch
                torchvision
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
