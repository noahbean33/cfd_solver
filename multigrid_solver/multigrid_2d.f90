! code is for educational purposes only
! two dimensional poisson equation using multigrid solver
! solves del^2 (phi)=-100; phi=zero on boundaries
! domain :   -1<=x<=1  and -1<=y<=1
! solution fixed to 0 on boundaries at i=1 and i=max; j=1 and j=jmax
! iterations over interior points from imin to imax and jmin to jmax
! centerline temperature series solution is 29.4685...
! possible grid points: 3,5,9,17,33,65,129,257,513 etc. in each direction
! level l=1 represents the finest mesh
! stride is a multiple of grid spacing; i.e., stride(l)=2*stride(l-1)

! must compile double precision using:  gfortran -O2 -fdefault-real-8 multigrid_2d.f90

! to run pure Gauss-Seidel w/o multigrid, specify 1 grid level and number of multigrid cycles
!                                    equal to the number of Gauss-Seidel iterations desired.
      MODULE multigrid_routines
      CONTAINS
!***************************************************************************************
      SUBROUTINE gauss_seidel(itermax,imin,imax,jmin,jmax,stride,h,l,x,b)
      IMPLICIT none
      REAL :: x(:,:,:),b(:,:,:),h(:)
      INTEGER :: stride(:),imin,imax,jmin,jmax,i,j,iter,l,itermax

      DO iter=1,itermax
      DO j=jmin,jmax,stride(l)
      DO i=imin,imax,stride(l)
      x(i,j,l)=0.25*(x(i+stride(l),j,l)+x(i-stride(l),j,l)+x(i,j+stride(l),l) + &
                             x(i,j-stride(l),l)) - h(l)**2*b(i,j,l)/4.0
      END DO; END DO; END DO
      RETURN
      END SUBROUTINE

!***************************************************************************************
!     compute error residuals and transfer down to coarser grids
      SUBROUTINE restriction(imin,imax,jmin,jmax,stride,h,l,x,b)
      IMPLICIT none
      REAL :: x(:,:,:),b(:,:,:),h(:)
      INTEGER :: stride(:),imin,imax,jmin,jmax,i,j,l
 
!     compute residuals
      DO i=imin,imax,stride(l)
      DO j=jmin,jmax,stride(l)
      b(i,j,l+1)=b(i,j,l) - (x(i+stride(l),j,l)+x(i-stride(l),j,l)+x(i,j+stride(l),l) + &
                                        x(i,j-stride(l),l) - 4.*x(i,j,l))/h(l)**2
      END DO;  END DO

      RETURN
      END SUBROUTINE

!***************************************************************************************
!     interpolate solutions from coarser to finer grids
      SUBROUTINE prolongation(imin,imax,jmin,jmax,stride,l,x)
      IMPLICIT none
      REAL :: x(:,:,:), correction
      INTEGER :: imin,imax,jmin,jmax,stride(:),l,i,j

!print*,l,stride,imin,imax,jmin,jmax
      ! 4 ave (circles on grid sketch)
!print*,'circles'    
      DO j=jmin,jmax,2*stride(l)
      DO i=imin,imax,2*stride(l)
       correction=0.25*(x(i-stride(l),j-stride(l),l+1)+x(i+stride(l),j+stride(l),l+1) + &
                x(i-stride(l),j+stride(l),l+1)+x(i+stride(l),j-stride(l),l+1))
!print*,i,j,correction
       x(i,j,l)=x(i,j,l)+correction
      END DO; END DO

      ! 2 ave (vertical average; triangles on grid sketch)
!print*,'triangles'
      DO j=jmin,jmax,2*stride(l)
      DO i=imin+stride(l),imax-stride(l),2*stride(l)
      correction=0.5*(x(i,j+stride(l),l+1)+x(i,j-stride(l),l+1))
!print*,i,j,correction
      x(i,j,l)=x(i,j,l)+correction
      END DO; END DO

      ! 2 ave (horizontal average; squares on grid sketch)
!print*,'squares'
      DO j=jmin+stride(l),jmax-stride(l),2*stride(l)
      DO i=imin,imax,2*stride(l)
      correction=0.5*(x(i+stride(l),j,l+1)+x(i-stride(l),j,l+1))
!print*,i,j,correction
      x(i,j,l)=x(i,j,l)+correction
      END DO; END DO
      
      ! 0 ave (no interpolation; diamonds on grid sketch)
!print*,'diamonds'
      DO j=jmin+stride(l),jmax-stride(l),2*stride(l)
      DO i=imin+stride(l),imax-stride(l),2*stride(l)
      correction=x(i,j,l+1)
!print*,i,j,correction
      x(i,j,l)=x(i,j,l)+correction
      END DO; END DO

      RETURN
      END SUBROUTINE
!***************************************************************************************
      END MODULE multigrid_routines

!***************************************************************************************     
      PROGRAM multigrid_2d
      USE multigrid_routines
      IMPLICIT NONE
      REAL,ALLOCATABLE :: x(:,:,:),b(:,:,:),h(:)
      INTEGER,ALLOCATABLE :: stride(:)
      INTEGER :: i,j,imin,imax,jmin,jmax,l,lmax,numouter,outer,max,itermax=1
      REAL :: rms,residual
      
      WRITE(*,*)'Enter the number of grid points in one direction'
      WRITE(*,'(a)',ADVANCE='NO')' i.e., 3,5,9,17,33,65,129,257,513,1025,2049:  '
      READ(*,*)max;  WRITE(*,*)
      WRITE(*,*)'Maximum number of multigrid levels is', NINT(LOG(REAL(max-1))/LOG(2.))
      WRITE(*,'(a)',ADVANCE='NO')' Enter number of grid levels:  '
      READ(*,*)lmax;  WRITE(*,*)
      WRITE(*,'(a)',ADVANCE='NO')' Enter the maximum number of multigrid cycles:  '
      READ(*,*)numouter;  WRITE(*,*)
      IF(lmax .gt. 1)THEN
        WRITE(*,'(a)',ADVANCE='NO')' Enter the number of Gauss-Seidel iterations per level:  '
        READ(*,*)itermax;  WRITE(*,*)
      END IF

      ALLOCATE(x(1:max,1:max,lmax),b(1:max,1:max,lmax),h(lmax),stride(lmax) )

!     initialize right side b to -100 in Ax=b
      b=-100; x=0.0; h(1)=2.0/REAL(max-1); stride(1)=1
      DO i=2,lmax
         h(i)=2.*h(i-1)
         stride(i)=2*stride(i-1)
      END DO

!*****************************************************************
      OUTERLOOP : DO outer=1,numouter
!     work down the cycle (smoothing or restriction)
!*****************************************************************
      x(:,:,2:lmax)=0.0     !initialize error solutions to zero
      DO l=1,lmax     ! cycle down through multigrid levels (restriction)
         imin=stride(l)+1; imax=max-stride(l); jmin=stride(l)+1; jmax=max-stride(l);
         CALL gauss_seidel(itermax,imin,imax,jmin,jmax,stride,h,l,x,b)
            IF(l .EQ. lmax)EXIT  !don't restrict to (nonexistent) lmax+1 grid level
            IF(lmax .EQ. 1)EXIT  !don't call restriction if no multigrid
         imin=stride(l+1)+1; imax=max-stride(l+1); jmin=stride(l+1)+1; jmax=max-stride(l+1);
         CALL restriction(imin,imax,jmin,jmax,stride,h,l,x,b)
      END DO

!******************************************************************
!     work up the cycle  (interpolation or prolongation)
      DO l=lmax-1,1,-1
         imin=stride(l)+1; imax=max-stride(l); jmin=stride(l)+1; jmax=max-stride(l);
         CALL prolongation(imin,imax,jmin,jmax,stride,l,x)
!    smoothing iterations of error on way up after interpolation (optional, but much faster)
         CALL gauss_seidel(5,imin,imax,jmin,jmax,stride,h,l,x,b)
      END DO

!******************************************************************
      rms=0.0  !compute top level l=1 rms residual
      DO i=2,max-1
      DO j=2,max-1
      residual=b(i,j,1) - (x(i+1,j,1)+x(i-1,j,1)+x(i,j+1,1)+x(i,j-1,1)- &
                                                  4.*x(i,j,1))/h(1)**2
      rms=rms+residual**2
      END DO;  END DO
      rms=SQRT(rms)
      WRITE(*,100)outer,rms,x((max+1)/2,(max+1)/2,1)
      IF(rms .LE. 1.e-5)EXIT      !exit loop if converged solution

      END DO OUTERLOOP

!     write solution to file for gnuplot contour 
      WRITE(*,*)'Writing data file ...'
      OPEN(UNIT=20,file='results.dat')     
      DO i=1,max
         WRITE(20,*)
      DO j=1,max
         WRITE(20,*)i,j,x(i,j,1)
      END DO; END DO
      WRITE(*,*)'Finished'

100   FORMAT(' After cycle',I5,'   RMS residual=',ES12.5,'   center point result=',ES12.5)

      END PROGRAM multigrid_2d

!***************************************************************************************