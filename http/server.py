from about import *
from io import BytesIO
from tqdm import tqdm
import html, mimetypes, os, posixpath, re, shutil
import http.server
import argparse, asyncio, sys
import urllib.request, urllib.parse, urllib.error


async def execute_run_script(input: str) -> None:
    RUN_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py', 'run.py'))
    with open(f'{os.path.splitext(input)[0]}-stdout.txt', 'w+') as fout:
        await asyncio.create_subprocess_exec(sys.executable, RUN_SCRIPT_PATH, '--input', input, stdout=fout, stderr=asyncio.subprocess.STDOUT)


class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        """Serve a GET request."""
        f = self.send_head()
        if f:
            self.copyfile(f, self.wfile)
            f.close()
 
    def do_HEAD(self):
        """Serve a HEAD request."""
        f = self.send_head()
        if f:
            f.close()
 
    def do_POST(self):
        """Serve a POST request."""
        r, info = self.deal_post_data()
        f = BytesIO()
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(b'<html>\n<title>Upload Result Page</title>\n')
        f.write(b'<body>\n<h2>Upload Result Page</h2>\n')
        f.write(b'<hr>\n')
        if r:
            f.write(b'<strong>Success:</strong>')
        else:
            f.write(b'<strong>Failed:</strong>')
        f.write(info.encode())
        f.write(f'<br><a href=\"{self.headers["referer"]}\">back</a>'.encode())
        f.write(b'<hr></body>\n</html>\n')
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-Length', str(length))
        self.end_headers()
        if f:
            self.copyfile(f, self.wfile)
            f.close()
        
    def deal_post_data(self):
        content_type = self.headers['content-type']
        if not content_type:
            return (False, 'Content-Type header doesn\'t contain boundary')
        boundary = content_type.split('=')[1].encode()
        remainbytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remainbytes -= len(line)
        if not boundary in line:
            return (False, 'Content NOT begin with boundary')
        line = self.rfile.readline()
        remainbytes -= len(line)
        fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line.decode())
        if not fn:
            return (False, 'Can\'t find out file name...')
        path = self.translate_path(self.path)
        fn = os.path.join(path, fn[0])
        pbar = tqdm(desc=f'Uploading video to "{fn}"', total=remainbytes, unit='B', unit_scale=True, unit_divisor=1024)
        line = self.rfile.readline()
        remainbytes -= len(line)
        pbar.update(len(line))
        line = self.rfile.readline()
        remainbytes -= len(line)
        pbar.update(len(line))
        try:
            out = open(fn, 'wb')
        except IOError:
            return (False, 'Can\'t create file to write, do you have permission to write?')
        preline = self.rfile.readline()
        remainbytes -= len(preline)
        pbar.update(len(preline))
        while remainbytes > 0:
            line = self.rfile.readline()
            remainbytes -= len(line)
            pbar.update(len(line))
            if boundary in line:
                preline = preline[0:-1]
                if preline.endswith(b'\r'):
                    preline = preline[0:-1]
                out.write(preline)
                out.close()
                pbar.close()
                asyncio.run(execute_run_script(fn))
                return (True, 'Successful upload. The result will be accessible once processing is complete. Click "Back" to see the current files on the server.')
            else:
                out.write(preline)
                preline = line
        return (False, 'Unexpect ends of data.')
 
    def send_head(self):
        """Common code for GET and HEAD commands.
        This sends the response code and MIME headers.
        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.
        """
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(301)
                self.send_header('Location', f'{self.path}/')
                self.end_headers()
                return None
            for index in ('index.html', 'index.htm'):
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            self.send_error(404, 'File not found')
            return None
        self.send_response(200)
        self.send_header('Content-type', ctype)
        fs = os.fstat(f.fileno())
        self.send_header('Content-Length', str(fs[6]))
        self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f
 
    def list_directory(self, path):
        """Helper to produce a directory listing (absent index.html).
        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().
        """
        try:
            VISIBLE_FILE_EXTENSIONS = ('.mp4', '.xlsx', '.txt')
            videos = list(filter(lambda arg: os.path.splitext(arg)[1].lower() in VISIBLE_FILE_EXTENSIONS, os.listdir(path)))
        except os.error:
            self.send_error(404, 'No permission to list directory')
            return None
        videos.sort(key=lambda a: a.lower())
        f = BytesIO()
        displaypath = html.escape(urllib.parse.unquote(self.path))
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(f'<html>\n<title>Directory listing for {displaypath}</title>\n'.encode())
        f.write(f'<body>\n<h2>Directory listing for {displaypath}</h2>\n'.encode())
        f.write(b'<hr>\n')
        f.write(b'<form ENCTYPE=\"multipart/form-data\" method=\"post\">')
        f.write(b'<input name=\"file\" type=\"file\" accept=\"video/*\">')
        f.write(b'<input type=\"submit\" value=\"upload\"/></form>\n')
        f.write(b'<hr>\n<ul>\n')
        for name in videos:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = f'{name}/'
                linkname = f'{name}/'
            if os.path.islink(fullname):
                displayname = f'{name}@'
                # Note: a link to a directory displays with @ and links with /
            f.write(f'<li><a href="{urllib.parse.quote(linkname)}">{html.escape(displayname)}</a>\n'.encode())
        f.write(b'</ul>\n<hr>\n</body>\n</html>\n')
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-Length', str(length))
        self.end_headers()
        return f
 
    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.
        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)
        """
        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        path = posixpath.normpath(urllib.parse.unquote(path))
        words = path.split('/')
        words = [_f for _f in words if _f]
        path = os.getcwd()
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir): continue
            path = os.path.join(path, word)
        return path
 
    def copyfile(self, source, outputfile):
        """Copy all data between two file objects.
        The SOURCE argument is a file object open for reading
        (or anything with a read() method) and the DESTINATION
        argument is a file object open for writing (or
        anything with a write() method).
        The only reason for overriding this would be to change
        the block size or perhaps to replace newlines by CRLF
        -- note however that this the default server uses this
        to copy binary data as well.
        """
        shutil.copyfileobj(source, outputfile)
 
    def guess_type(self, path):
        """Guess the type of a file.
        Argument is a PATH (a filename).
        Return value is a string of the form type/subtype,
        usable for a MIME Content-type header.
        The default implementation looks the file's extension
        up in the table self.extensions_map, using application/octet-stream
        as a default; however it would be permissible (if
        slow) to look inside the data to make a better guess.
        """
        _, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']
 
    if not mimetypes.inited:
        mimetypes.init() # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream', # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
        '.txt': 'text/plain',
        '.mp4': 'video/mp4',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    })
 
 
if __name__ == '__main__':
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', metavar='ADDRESS', type=str, default='localhost', help='the server address')
    parser.add_argument('--port', metavar='PORT', type=int, default=8000, help='the desired port number')
    # Parse arguments.
    args = parser.parse_args()
    # Start the server.
    httpd = http.server.HTTPServer((args.address, args.port), SimpleHTTPRequestHandler)
    address, port = httpd.server_address
    print(f'Start serving HTTP on {address} port {port} (http://{address}:{port}/)...', flush=True)
    httpd.serve_forever()
