!@file m_burger
!@brief Module that contains FWD, TLM and ADJ for 1D burger equations.

!@details Module that contains FWD, TLM and ADJ for 1D burger equations.
! the original author is Seon Ki Park (03/24/98). Updated to the
! Fortran-90 style.

module m_burger
  implicit none

  private
  public :: burger, burger_tlm, burger_adj

  real, parameter :: R = 100.d0          ! Reynolds number (reciprocal of diffusion coefficient)
  real, parameter :: dx = 1.d0           ! space increment
  real, parameter :: dt = 0.1d0          ! time increment
  real, parameter :: dtdx = dt/dx
  real, parameter :: dtdxsq = dt/(dx**2)
  real, parameter :: c1 = (2.d0/R)*dtdxsq
  real, parameter :: c0 = 1.d0/(1.d0+c1)
contains

  subroutine burger(nx, n, ui, uob, u, cost)
    implicit none
    integer, intent(in) :: nx             ! # of grid pts
    integer, intent(in) :: n              ! # of time steps
    real, intent(in) :: uob(nx, n)        ! observations
    real, intent(inout) :: ui(nx)         ! initial conditions
    real, intent(out) :: u(nx, n)         ! model solutions
    real, intent(out) :: cost             ! cost function

    integer :: i, j

    ! initialize the cost function:
    cost = 0.d0

    ! set the initial conditions:
    do i = 1, nx
      u(i,1) = ui(i)
    enddo

    ! set the boundary conditions:
    do j = 1, n
      u(1, j) = 0.d0
      u(nx, j) = 0.d0
    enddo

    ! integrate the model numerically:
    ! FTCS for the 1st time step integration
    do i = 2, nx-1
      u(i,2) = u(i,1) - 0.5d0*dtdx*u(i,1)*(u(i+1,1) - u(i-1,1)) &
              + 0.5d0*c1*(u(i+1,1) - 2.d0*u(i,1) + u(i-1,1))
    enddo

    ! Leapfrog/DuFort–Frankel afterwards
    do j = 3, n
      do i = 2, nx-1
        u(i,j) = c0*(u(i,j-2) + c1*(u(i+1,j-1) - u(i,j-2) + u(i-1,j-1))) &
                - dtdx*u(i,j-1)*(u(i+1,j-1) - u(i-1,j-1))
      enddo
    enddo

    ! cost function:
    do j = 1, n
      do i = 1, nx
        cost = cost + 0.5*(u(i,j) - uob(i,j))**2
      enddo
    enddo

    ! save nonlinear solutions to the basic fields:
    do j = 1, n
      do i = 1, nx
        uob(i,j) = u(i,j)
      enddo
    enddo
  end subroutine burger

  subroutine burger_tlm(nx, n, ui, ubasic, u)
    ! The tangent linear model of the burger’s equations
    implicit none

    integer, intent(in) :: nx
    integer, intent(in) :: n
    real, intent(in) :: ui(nx)           ! initial conditions
    real, intent(in) :: ubasic(nx, n)    ! basic states
    real, intent(out) :: u(nx, n)        ! TLM solutions

    integer :: i, j

    ! set the initial conditions:
    do i = 1, nx
      u(i,1) = ui(i)
    enddo

    ! set the boundary conditions:
    do j = 1, n
      u(1, j) = 0.d0
      u(nx, j) = 0.d0
    enddo

    ! FTCS for 1st time step integration
    do i = 2, nx-1
      u(i,2) = u(i,1) - 0.5d0*dtdx*(ubasic(i,1)*(u(i+1,1) - u(i-1,1)) &
             + u(i,1)*(ubasic(i+1,1) - ubasic(i-1,1))) &
             + 0.5d0*c1*(u(i+1,1) - 2.d0*u(i,1) + u(i-1,1))
    enddo

    ! leap frog / DuFort–Frankel afterwards
    do j = 3, n
      do i = 2, nx-1
        u(i,j) = c0*(u(i,j-2) + c1*(u(i+1,j-1) - u(i,j-2) + u(i-1,j-1))) &
               - dtdx*(u(i,j-1)*(ubasic(i+1,j-1) - ubasic(i-1,j-1)) &
               + ubasic(i,j-1)*(u(i+1,j-1) - u(i-1,j-1)))
      enddo
    enddo

    ! set the final value of u in ui:
    do i = 1, nx
      ui(i) = u(i,n)
    enddo
  end subroutine burger_tlm

  subroutine burger_adj(iforcing, nx, n, ui, ubasic, uob, u)
    ! The adjoint model of the burger’s equations
    implicit none

    integer, intent(in) :: iforcing
    integer, intent(in) :: nx
    integer, intent(in) :: n
    real, intent(inout) :: ui(nx)
    real, intent(inout) :: ubasic(nx, n)
    real, intent(inout) :: uob(nx, n)
    real, intent(inout) :: u(nx, n)

    integer :: i, j

    ! initialize adjoint variables:
    do j = 1, n
      do i = 1, nx
        u(i, j) = 0.d0
      enddo
    enddo

    ! set the final conditions:
    if (iforcing .eq. 0) then
      do i = 1, nx
        u(i, n) = ui(i)
        ui(i) = 0.d0
      enddo
    else
      do i = 1, nx
        u(i, n) = ubasic(i, n) - uob(i, n)
        ui(i) = 0.d0
      enddo
    endif

    ! adjoint of Leap Frog / DuFort–Frankel
    do j = n, 3, -1
      do i = nx-1, 2, -1
        u(i,j-2) = u(i,j-2) + c0*(c1*(u(i+1,j-1) - u(i-1,j-1)) &
                 - dtdx*ubasic(i,j-1)*(u(i+1,j-1) - u(i-1,j-1))) &
                 + ubasic(i,j-1)*(u(i+1,j-1) - 2.d0*u(i,j-1) + u(i-1,j-1))
      enddo
    enddo

    if (iforcing .eq. 1) then
      do i = 1, nx
        u(i,1) = u(i,1) + ubasic(i,1) - uob(i,1)
      enddo
    endif

    ! adjoint of FTCS
    do i = nx-1, 2, -1
      u(i-1,1) = u(i-1,1) + 0.5d0*(c1+dtdx*ubasic(i,1))*u(i,2)
      u(i,1)   = u(i,1)   + (1.d0 - c1)*u(i,2)
      u(i+1,1) = u(i+1,1) + 0.5d0*(c1 - dtdx*ubasic(i,1))*u(i,2)
    enddo

    if (iforcing .eq. 1) then
      do i = 1, nx
        ui(i) = ui(i) + ubasic(i,1) - uob(i,1)
      enddo
    endif

    ! set the boundary conditions:
    do j = 1, n
      u(1, j) = 0.d0
      u(nx, j) = 0.d0
    enddo

    ! set the final value of u in ui:
    do i = 1, nx
      ui(i) = ui(i) + u(i,1)
    enddo

  end subroutine burger_adj

end module m_burger
