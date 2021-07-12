import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter.filedialog import askopenfilenames, askdirectory
from functools import partial
import os
import threading
import sys
import logging
import logging.handlers
import pathlib
import webbrowser

from . import logging as logutils
from ..version import version
from .. import AA_stat, io

AA_STAT_VERSION = version

INPUT_FILES = []
INPUT_SPECTRA = []
OUTDIR = '.'


class Args:
    """Emulates parsed args from argparse for AA_stat"""
    pepxml = mgf = mzml = csv = params = None
    dir = '.'
    verbosity = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_input_filenames(label, activate):
    fnames = askopenfilenames(title='Open search results',
        filetypes=[('pepXML file', '*.pepXML'), ('XML file', '*.pep.xml'), ('CSV file', '*.[ct]sv')],
        multiple=True)

    if fnames:
        INPUT_FILES[:] = fnames
        label['text'] = f'{len(fnames)} open search files selected.'  #+ '\n'.join(os.path.basename(f) for f in fnames)
        activate['state'] = tk.NORMAL


def get_spectrum_filenames(label):
    fnames = askopenfilenames(title='Spectrum files',
        filetypes=[('mzML file', '*.mzML'), ('MGF file', '*.mgf')],
        multiple=True)

    if fnames:
        INPUT_SPECTRA[:] = fnames
        label['text'] = f'{len(fnames)} spectrum files selected.'  # + '\n'.join(os.path.basename(f) for f in fnames)


def get_outdir_name(label):
    global OUTDIR
    dirname = askdirectory(title='Output directory')
    if dirname:
        OUTDIR = dirname
    label['text'] = 'Output directory: ' + os.path.abspath(dirname)


def get_aa_stat_version():
    if AA_STAT_VERSION:
        return 'AA_stat v' + AA_STAT_VERSION
    else:
        return 'AA_stat not installed.'


def get_aa_stat_args():
    pepxml, csv = [], []
    for f in INPUT_FILES:
        ext = os.path.splitext(f)[1].lower()
        if ext in {'.pepxml', '.xml'}:
            pepxml.append(f)
        else:
            csv.append(f)
    mzml, mgf = [], []
    for f in INPUT_SPECTRA:
        ext = os.path.splitext(f)[1].lower()
        if ext == '.mzml':
            mzml.append(f)
        else:
            mgf.append(f)
    args = Args(pepxml=pepxml, mgf=mgf, csv=csv, mzml=mzml, dir=OUTDIR)
    params_dict = io.get_params_dict(args)
    return args, params_dict


def start_aastat(t):
    t.start()


def run_aastat(run_btn, status_to, log_to):
    run_btn['state'] = tk.DISABLED
    status_to['text'] = 'Checking arguments...'
    args, params_dict = get_aa_stat_args()
    status_to['text'] = 'Running AA_stat...'

    AA_stat.AA_stat(params_dict, args)
    status_to['text'] = 'Done.'
    run_btn['state'] = tk.NORMAL
    run_btn['text'] = 'View report'
    run_btn['command'] = partial(view_report, run_btn)


def view_report(btn):
    url = (pathlib.Path(os.path.abspath(OUTDIR)) / 'report.html').as_uri()
    webbrowser.open(url)


def main():
    window = tk.Tk()
    window.title('AA_stat GUI')
    window.geometry('900x600')
    try:
        try:
            window.tk.call('tk_getOpenFile', '-foobarbaz')
        except tk.TclError:
            pass

        window.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
        window.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
    except:
        pass

    top_frame = tk.Frame()
    input_frame = tk.Frame(master=top_frame)

    spectra_frame = tk.Frame(master=top_frame)
    selected_spectra_lbl = tk.Label(master=spectra_frame, text="(optional)", justify='left')
    get_spectra_btn = tk.Button(master=spectra_frame, text="Select spectrum files",
        command=partial(get_spectrum_filenames, selected_spectra_lbl), width=20)

    get_spectra_btn.pack(side=tk.LEFT, fill=tk.X, padx=15, anchor=tk.E)
    selected_spectra_lbl.pack(side=tk.LEFT, fill=tk.X, anchor=tk.W)

    dir_frame = tk.Frame(master=top_frame)
    dir_lbl = tk.Label(master=dir_frame, text="Output directory: " + os.path.abspath(OUTDIR), justify='left')
    get_dir_btn = tk.Button(master=dir_frame, text="Select output directory",
        command=partial(get_outdir_name, dir_lbl), width=20)

    get_dir_btn.pack(side=tk.LEFT, padx=15, fill=tk.X, anchor=tk.E)
    dir_lbl.pack(side=tk.LEFT, fill=tk.X, anchor=tk.W)

    main_frame = tk.Frame()
    run_btn = tk.Button(master=main_frame, text='Run AA_stat', state=tk.DISABLED)

    status_lbl = tk.Label(master=main_frame, text=get_aa_stat_version())
    log_txt = ScrolledText(master=main_frame, state=tk.DISABLED)
    t = threading.Thread(target=run_aastat, args=(run_btn, status_lbl, log_txt), name='aastat-runner')
    t.daemon = True
    run_btn['command'] = partial(start_aastat, t)


    logger = logutils.get_logger()
    AAstatHandler = logutils.get_aastat_handler(log_txt)

    log_t = threading.Thread(target=logutils._socket_listener_worker,
        args=(logger, logging.handlers.DEFAULT_TCP_LOGGING_PORT, AAstatHandler),
        name='aastat-listener')
    log_t.start()
    logger.debug('AA_stat logging initiated.')


    log_txt.pack(fill=tk.BOTH, expand=True)
    run_btn.pack()
    status_lbl.pack()

    selected_os_lbl = tk.Label(master=input_frame, text="No files selected", justify='left')

    get_os_files_btn = tk.Button(master=input_frame, text="Select open search files",
        command=partial(get_input_filenames, selected_os_lbl, run_btn), width=20)

    get_os_files_btn.pack(side=tk.LEFT, padx=15, fill=tk.X, anchor=tk.E)
    selected_os_lbl.pack(side=tk.LEFT, fill=tk.X, anchor=tk.W)


    input_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
    spectra_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
    dir_frame.pack(side=tk.TOP, fill=tk.X, expand=True)

    top_frame.pack()
    main_frame.pack(fill=tk.BOTH, expand=True)
    if not AA_STAT_VERSION:
        for btn in [get_spectra_btn, get_os_files_btn, get_dir_btn]:
            btn['state'] = tk.DISABLED
    window.mainloop()
    logutils.tcpserver.abort = 1
    logutils.tcpserver.server_close()
    sys.exit()  # needed because there are still working (daemon) threads


if __name__ == '__main__':
	main()
