! ONE DIMENSIONAL MULTIGRID CODE
! code is for educational purposes only
! solves d^2(phi)/dx^2=0
! solution fixed to zero on boundaries at i=1 and i=max
! possible grid points: 3,5,9,17,33,65,129,257,513,1025,etc.(these include boundary points)
! iterations over interior points from imin to imax
! lmax is number of grid levels (1 is no refinement, Gauss-Seidel solver)
! max is maximum number of grid points on fine grid
! level l=1 represents the finest mesh
! stride is a multiple of grid spacing; i.e., stride(l)=2*stride(l-1)

! compile double precision with gfortran -fdefault-real-8 -O2 multigrid_1d.f90
! to run pure Gauss-Seidel w/o multigrid, specify 1 grid level and number of multigrid cycles
!                                    equal to the number of Gauss-Seidel iterations desired.
      MODULE multigrid_routines
      CONTAINS
!**********************************************************************
      SUBROUTINE gauss_seidel(itermax,imin,imax,stride,h,l,x,b)
      IMPLICIT none
      REAL :: x(:,:),b(:,:), h(:)
      INTEGER :: stride(:),imin,imax,i,l,itermax,iter

      DO iter=1,itermax
      DO i=imin,imax,stride(l)
         x(i,l)=( x(i+stride(l),l)+x(i-stride(l),l) - h(l)**2*b(i,l) )/2.0
      END DO; END DO
      RETURN
      END SUBROUTINE

!***********************************************************************
!     compute error residuals and transfer down to coarser grids
      SUBROUTINE restriction(imin,imax,stride,h,l,x,b)
      IMPLICIT none
      REAL :: x(:,:),b(:,:),h(:)
      INTEGER :: stride(:),imin,imax,i,l

!     note b(i,l+1) is the residual vector r for solving Ae=r on next coarse level
!     direct transfer on down to coarser grid; no interpolation necessary
      DO i=imin,imax,stride(l+1)
         b(i,l+1)=b(i,l) - ( x(i+stride(l),l)+x(i-stride(l),l) - 2.*x(i,l) )/h(l)**2
      END DO
     
      RETURN
      END SUBROUTINE

!***************************************************************************************
!     interpolate errors up from coarser grids to finer grids
      SUBROUTINE prolongation(imin,imax,stride,l,x)
      IMPLICIT none
      REAL :: x(:,:),correction
      INTEGER :: imin,imax,stride(:),l,i

!     direct transfer of error up across grids
      DO i=imin+stride(l),imax-stride(l),2*stride(l)
          correction=x(i,l+1)
          x(i,l)=x(i,l)+correction
      END DO
!     linear interpolation of error between points moving to finer grid
      DO i=imin,imax,2*stride(l)
          correction=0.5*(x(i+stride(l),l+1)+x(i-stride(l),l+1))
          x(i,l)=x(i,l)+correction
      END DO
      RETURN
      END SUBROUTINE

!***************************************************************************************
      END MODULE multigrid_routines

!***************************************************************************************     
      PROGRAM multigrid
      USE multigrid_routines
      IMPLICIT none
      REAL, PARAMETER :: pi=4.0*ATAN(1.0)
      REAL, ALLOCATABLE :: x(:,:),b(:,:),h(:)
      INTEGER, ALLOCATABLE :: stride(:)
      INTEGER :: imin,imax,l,lmax,numouter,outer,i,max,itermax=1
      INTEGER (KIND=8) :: start,end,countrate
      REAL :: walltime,ymax,residual,rms

!     set some parameters
      WRITE(*,*)'Enter the number of grid points in one direction'
      WRITE(*,'(a)',ADVANCE='NO')'i.e., 3,5,9,17,33,65,129,257,513,1025,2049,4097:  '
      READ(*,*)max;  WRITE(*,*)
      WRITE(*,*)'Maximum number of multigrid levels is:', NINT(LOG(REAL(max-1))/LOG(2.))
      WRITE(*,'(a)',ADVANCE='NO')'Enter number of grid levels:  '
      READ(*,*)lmax;  WRITE(*,*)
      WRITE(*,'(a)',ADVANCE='NO')'Enter the maximum number of multigrid cycles:  '
      READ(*,*)numouter;  WRITE(*,*)
      IF(lmax .GT. 1) THEN
        WRITE(*,'(a)',ADVANCE='NO')'Enter the number of Gauss-Seidel iterations per level:  '
        READ(*,*)itermax;  WRITE(*,*)
      END IF
      ALLOCATE(x(1:max,lmax),b(1:max,lmax),h(lmax),stride(lmax))
     
      IF(lmax .eq. 1)then
         OPEN(UNIT=10,FILE='solution.gs')
         OPEN(UNIT=11,FILE='residual.gs')
         OPEN(UNIT=12,FILE='error.gs')
      ELSE
         OPEN(UNIT=10,FILE='solution.mg')
         OPEN(UNIT=11,FILE='residual.mg')
         OPEN(UNIT=12,FILE='error.mg')
      END IF        

      ymax=1.0            !domain length
      b=0.                !set right hand side of Ax=b
      h(1)=ymax/(max-1)   !grid spacing on finest mesh, h
      stride(1)=1         !initialize the stride at one on finest mesh spacing h

      DO i=2,lmax         !compute coarser grid spacings and strides
        h(i)=2.*h(i-1)
        stride(i)=2*stride(i-1)
      END DO

!     set initial condition on dependent variable on fine grid
      DO i=1,max
          x(i,1)=sin( 10.*(i-1)*pi/(max-1.0))
      END DO
      x(1,:)=0.0; x(max,:)=0.0    !(reset) boundary conditions

      call system_clock(start,countrate)

      OUTERLOOP : DO outer=1,numouter
!*******************************************************************
!     work down the cycle
      x(:,2:lmax)=0    ! initialize the error solutions to zero
      DO l=1,lmax     ! cycle through multigrid levels
            imin=stride(l)+1; imax=max-stride(l)
         CALL gauss_seidel(itermax,imin,imax,stride,h,l,x,b)
            IF(l .EQ. lmax)EXIT  !don't restrict to lmax+1 grid level
            IF(lmax .EQ. 1)EXIT  !don't call restriction if no multigrid
            imin=stride(l+1)+1; imax=max-stride(l+1)
         CALL restriction(imin,imax,stride,h,l,x,b)
      END DO
!********************************************************************
!     work up the cycle by interpolating errors to finer meshes

      DO l=lmax-1,1,-1  !note loop will not proceed if lmax=1; i.e., no multigrid
           imin=stride(l)+1; imax=max-stride(l)
           CALL prolongation(imin,imax,stride,l,x)
!     smoothing iterations of error on way up after interpolation (optional, but much faster)
           CALL gauss_seidel(3,imin,imax,stride,h,l,x,b)  
      END DO

!**********************************************************************
!     compute and write RMS residuals on finest grid
      rms=0.0;   !initialize to zero
      DO i=2,max-1
         residual=b(i,1) - ( x(i+1,1)+x(i-1,1) - 2.*x(i,1) )/h(1)**2
         rms=rms+residual**2
      END DO
      rms=SQRT(rms)
      WRITE(*,100)outer,rms,MAXVAL(ABS(x(:,1)))
      WRITE(11,*)outer,rms              !write RMS error
      WRITE(12,*)outer,MAXVAL(x(:,1))   !write maximum error if solution is zero
      IF(rms .LE. 1.e-07)EXIT           !stop if converged solution
      END DO OUTERLOOP

!     write solution to file
      DO i=1,max
         WRITE(10,*)i,x(i,1)
      END DO
!**********************************************************************
      PRINT*; PRINT*
100   FORMAT(' After cycle',I7,'   RMS residual=',ES12.5,'   maximum error=',ES12.5)
      CALL system_clock(end)
      walltime=REAL(end-start)/REAL(countrate)
      PRINT*,'Elapsed time= ',walltime;   PRINT*
      PRINT*,'Solution written to solution.xx'
      PRINT*,'RMS residuals written to residual.xx'
      PRINT*,'Solution errors written to error.xx'
      PRINT*

      END PROGRAM multigrid
!**********************************************************************