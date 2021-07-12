import logging
import logging.handlers

import socketserver
import struct
import pickle
import socket
import tkinter as tk


class LoggingToGUI(logging.Handler):
    # https://stackoverflow.com/a/18194597/1258041
    def __init__(self, console):
        logging.Handler.__init__(self)
        self.console = console

    def emit(self, message):
        formattedMessage = self.format(message)

        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, formattedMessage + '\n')
        self.console.configure(state=tk.DISABLED)
        self.console.see(tk.END)


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request."""

    def __init__(self, *args, **kwargs):
        socketserver.StreamRequestHandler.__init__(self, *args, **kwargs)

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        self._record_handler.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


tcpserver = None
def _socket_listener_worker(logger, port, handler):
    global tcpserver
    try:
        tcpserver = LogRecordSocketReceiver(port=port, handler=handler)
    except socket.error as e:
        logger.error('Couldn\'t start TCP server: %s', e)
        return
    if port == 0:
        port = tcpserver.socket.getsockname()[1]
    tcpserver.serve_until_stopped()


def get_logger():
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    formatter = logging.Formatter('{levelname:>8}: {asctime} {message}',
                datefmt='[%H:%M:%S]', style='{')
    stream_handler.setFormatter(formatter)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    tcpHandler = logging.handlers.SocketHandler('localhost', logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    tcpHandler.setLevel(logging.INFO)
    logging.getLogger('AA_stat').addHandler(tcpHandler)
    return logger


def get_aastat_handler(log_txt):
    class AAstatHandler(LogRecordStreamHandler):
        def __init__(self, *args, **kwargs):
            self._record_handler = LoggingToGUI(log_txt)
            formatter = logging.Formatter('{levelname:>8}: {asctime} {message}',
                datefmt='[%H:%M:%S]', style='{')
            self._record_handler.setFormatter(formatter)
            super().__init__(*args, **kwargs)

    return AAstatHandler
