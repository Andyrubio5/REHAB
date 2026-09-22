"""Ejecutar desde la raíz: python -m rehab --help."""
import argparse
from pathlib import Path
import nbformat
from .data import ROOT, load_data, validate_frame, verify_manifest, rebuild, compare_rebuilt, sha256
from .experiments import run, fit_final, save_json, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Verifica el CSV, hashes y formato de notebooks.')
    validate.add_argument('--arrays', action='store_true')
    build = commands.add_parser('rebuild', help='Reconstruye sin sobrescribir el CSV publicado.')
    build.add_argument('--output', default='reports/runs/rebuilt.csv')
    for stage in ('selection', 'search'):
        sub = commands.add_parser(stage)
        sub.add_argument('--mode', choices=['rapido', 'completo'], default='rapido')
        sub.add_argument('--output', type=Path)
    final = commands.add_parser('fit-final')
    final.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'validate':
        verify_manifest(args.arrays)
        print(validate_frame(load_data()))
        for path in sorted(ROOT.glob('*.ipynb')):
            nbformat.validate(nbformat.read(path, as_version=4))
            print('Notebook válido:', path.name)
    elif args.command == 'rebuild':
        verify_manifest(include_arrays=True)
        df = rebuild()
        difference = compare_rebuilt(df)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f'No se sobrescribe {output}; elige otra salida.')
        df.to_csv(output, index=False)
        save_json(output.with_suffix('.provenance.json'), {'max_abs_difference': difference,
                  'generated_sha256': sha256(output), 'environment': environment(),
                  'input_manifest': verify_manifest(True)})
        print(f'Reconstrucción verificada: {output}; diferencia máxima={difference}')
    elif args.command == 'fit-final':
        print(fit_final(args.run))
    else:
        print(run(args.command, args.mode, args.output))


if __name__ == '__main__':
    main()
