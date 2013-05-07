MODULE cufor_mpi_mod

  USE data_module
  USE definitions_module
  USE MPI

  IMPLICIT NONE

CONTAINS

  SUBROUTINE cufor_mpi_interop &
  (chunk,       &
  which_array,    &
  left_snd_buffer,  &
  left_rcv_buffer,  &
  right_snd_buffer,   &
  right_rcv_buffer,   &
  bottom_snd_buffer,  &
  bottom_rcv_buffer,  &
  top_snd_buffer,   &
  top_rcv_buffer,   &
  depth,        &
  field_type      )

    IMPLICIT NONE

    INTEGER :: which_array, chunk, depth, field_type
    REAL(KIND=8) :: left_snd_buffer(:),left_rcv_buffer(:),right_snd_buffer(:),right_rcv_buffer(:)
    REAL(KIND=8) :: bottom_snd_buffer(:),bottom_rcv_buffer(:),top_snd_buffer(:),top_rcv_buffer(:)

    INTEGER :: tag, x_inc, y_inc, receiver, message_count, err, sender
    INTEGER :: request(8)
    INTEGER :: status(MPI_STATUS_SIZE, 8)

    ! buffer sizes
    INTEGER :: left_buf_sz, right_buf_sz, top_buf_sz, bottom_buf_sz

    ! which face to copy
    INTEGER :: LEFT_FACE=0, RIGHT_FACE=1, TOP_FACE=2, BOTTOM_FACE=3

    IF(field_type.EQ.CELL_DATA) THEN
      x_inc=0
      y_inc=0
    ENDIF
    IF(field_type.EQ.VERTEX_DATA) THEN
      x_inc=1
      y_inc=1
    ENDIF
    IF(field_type.EQ.X_FACE_DATA) THEN
      x_inc=1
      y_inc=0
    ENDIF
    IF(field_type.EQ.Y_FACE_DATA) THEN
      x_inc=0
      y_inc=1
    ENDIF

    ! set to col/row size
    left_buf_sz = (chunks(chunk)%field%y_max+5)*depth
    right_buf_sz = (chunks(chunk)%field%y_max+5)*depth

    top_buf_sz = (chunks(chunk)%field%x_max+5)*depth
    bottom_buf_sz = (chunks(chunk)%field%x_max+5)*depth

    request=0
    message_count=0

    IF(parallel%task.EQ.chunks(chunk)%task) THEN
      IF(chunks(chunk)%chunk_neighbours(chunk_left).NE.external_face) THEN
        tag=4*(chunk)+1
        receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_left))%task

        !PACK BUFFER
        CALL cudapackbuffers(which_array, LEFT_FACE, left_snd_buffer, left_buf_sz, depth)

        CALL MPI_ISEND(left_snd_buffer,left_buf_sz,MPI_DOUBLE_PRECISION,receiver,tag &
              ,MPI_COMM_WORLD,request(message_count+1),err)
        tag=4*(chunks(chunk)%chunk_neighbours(chunk_left))+2
        sender=chunks(chunks(chunk)%chunk_neighbours(chunk_left))%task
        CALL MPI_IRECV(left_rcv_buffer,left_buf_sz,MPI_DOUBLE_PRECISION,sender,tag &
              ,MPI_COMM_WORLD,request(message_count+2),err)
        message_count=message_count+2
      ENDIF

      IF(chunks(chunk)%chunk_neighbours(chunk_right).NE.external_face) THEN

        tag=4*chunk+2 ! 4 because we have 4 faces, 2 because it is leaving the right face
        receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_right))%task

        !PACK BUFFER
        CALL cudapackbuffers(which_array, RIGHT_FACE, right_snd_buffer, right_buf_sz, depth)

        CALL MPI_ISEND(right_snd_buffer,right_buf_sz,MPI_DOUBLE_PRECISION,receiver,tag &
              ,MPI_COMM_WORLD,request(message_count+1),err)
        tag=4*(chunks(chunk)%chunk_neighbours(chunk_right))+1 ! 4 because we have 4 faces, 1 because it is coming from the left face of the right neighbour
        sender=chunks(chunks(chunk)%chunk_neighbours(chunk_right))%task
        CALL MPI_IRECV(right_rcv_buffer,right_buf_sz,MPI_DOUBLE_PRECISION,sender,tag, &
               MPI_COMM_WORLD,request(message_count+2),err)
        message_count=message_count+2
      ENDIF
    ENDIF

    CALL MPI_WAITALL(message_count,request,status,err)

    IF(chunks(chunk)%chunk_neighbours(chunk_left).NE.external_face) THEN
      call cudaunpackbuffers(which_array, LEFT_FACE, left_rcv_buffer, left_buf_sz, depth)
    ENDIF
    IF(chunks(chunk)%chunk_neighbours(chunk_right).NE.external_face) THEN
      call cudaunpackbuffers(which_array, RIGHT_FACE, right_rcv_buffer, right_buf_sz, depth)
    ENDIF

    request=0
    message_count=0


    IF(parallel%task.EQ.chunks(chunk)%task) THEN
      IF(chunks(chunk)%chunk_neighbours(chunk_bottom).NE.external_face) THEN
        tag=4*(chunk)+3 ! 4 because we have 4 faces, 3 because it is leaving the bottom face
        receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_bottom))%task

        !PACK BUFFER
        CALL cudapackbuffers(which_array, BOTTOM_FACE, bottom_snd_buffer, bottom_buf_sz, depth)

        CALL MPI_ISEND(bottom_snd_buffer,bottom_buf_sz,MPI_DOUBLE_PRECISION,receiver,tag &
              ,MPI_COMM_WORLD,request(message_count+1),err)
        tag=4*(chunks(chunk)%chunk_neighbours(chunk_bottom))+4 ! 4 because we have 4 faces, 1 because it is coming from the top face of the bottom neighbour
        sender=chunks(chunks(chunk)%chunk_neighbours(chunk_bottom))%task
        CALL MPI_IRECV(bottom_rcv_buffer,bottom_buf_sz,MPI_DOUBLE_PRECISION,sender,tag &
              ,MPI_COMM_WORLD,request(message_count+2),err)
        message_count=message_count+2
      ENDIF

      IF(chunks(chunk)%chunk_neighbours(chunk_top).NE.external_face) THEN
        tag=4*(chunk)+4 ! 4 because we have 4 faces, 4 because it is leaving the top face
        receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_top))%task

        !PACK BUFFER
        CALL cudapackbuffers(which_array, TOP_FACE, top_snd_buffer, top_buf_sz, depth)

        CALL MPI_ISEND(top_snd_buffer,top_buf_sz,MPI_DOUBLE_PRECISION,receiver,tag &
              ,MPI_COMM_WORLD,request(message_count+1),err)
        tag=4*(chunks(chunk)%chunk_neighbours(chunk_top))+3 ! 4 because we have 4 faces, 4 because it is coming from the left face of the top neighbour
        sender=chunks(chunks(chunk)%chunk_neighbours(chunk_top))%task
        CALL MPI_IRECV(top_rcv_buffer,top_buf_sz,MPI_DOUBLE_PRECISION,sender,tag, &
               MPI_COMM_WORLD,request(message_count+2),err)
        message_count=message_count+2
      ENDIF
    ENDIF

    ! Wait for the messages

    CALL MPI_WAITALL(message_count,request,status,err)

    IF(chunks(chunk)%chunk_neighbours(chunk_top).NE.external_face) THEN
      call cudaunpackbuffers(which_array, TOP_FACE, top_rcv_buffer, top_buf_sz, depth)
    ENDIF
    IF(chunks(chunk)%chunk_neighbours(chunk_bottom).NE.external_face) THEN
      call cudaunpackbuffers(which_array, BOTTOM_FACE, bottom_rcv_buffer, bottom_buf_sz, depth)
    ENDIF

  END SUBROUTINE cufor_mpi_interop

END MODULE cufor_mpi_mod

