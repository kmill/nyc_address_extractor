import tornado
import tornado.web
import tornado.httpserver
import tornado.ioloop
import os.path
import json

class ExtractHandler(tornado.web.RequestHandler) :
    def get(self) :
        reload(detect2)
        req = json.loads(self.get_argument('req'))
        if req['type'] == 'location' :
            res = detect2.determine_locations(req['text'])
        elif req['type'] == 'signs' :
            res = {"text" : detect2.sign_desc_for_locations(req['locs'])}
        elif req['type'] == 'addresses' :
            res = detect2.determine_address(req['text'])
        self.write(json.dumps({'response' : res}))

class Application(tornado.web.Application) :
    def __init__(self) :
        settings = {
            "static_path" : os.path.join(os.path.dirname(__file__), "static"),
            }
        handlers = [
            (r'/extract', ExtractHandler)
            ]
        tornado.web.Application.__init__(self, handlers, **settings)

print "importing detect2..."
import detect2
print "finished loading detect2."

if __name__ == "__main__" :
    application = Application()
    application.listen(8080)
    print "Running..."
    tornado.ioloop.IOLoop.instance().start()
