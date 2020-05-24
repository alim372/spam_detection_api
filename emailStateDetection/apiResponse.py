
def prepareResponse(meta=[], permissions=[], data=[], status=True, message='', errors=[]):
    response = {
        'meta' :meta,
        'permissions':permissions,
        'data': data ,
        'status': status ,
        'message':message ,
        'errors':errors
    }
    return response