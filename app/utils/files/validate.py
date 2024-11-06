from fastapi import HTTPException, UploadFile, status

def validate_file_size_type(file: UploadFile) -> None:
    FILE_SIZE = 10485760 # 10MB

    try:
        if file.filename.split('.')[-1] not in ['png', 'jpeg', 'jpg'] or file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            print('Error unsupported file typpe', file.filename.split('.')[-1])
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail='Unsupported file type',
            )
        
        real_file_size = 0
        for chunk in file.file:
            real_file_size += len(chunk)
            if real_file_size > FILE_SIZE:
                print('Error file is too large:', real_file_size, 'B')
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
                    detail='File is too large'
                )
            
        file.file.seek(0) # Seek to the beginning of the file object

    except Exception as error:
        print('Error occured when validating file :', error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Something went wrong when validating file'
        )