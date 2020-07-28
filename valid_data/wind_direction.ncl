begin
;diri = "/home/liyuan3970/Data/data/meto_data/radar_typhoon_liqima/wenzhou_rada/data_120/"
diri = "/home/liyuan3970/Data/data/meto_data/radar_typhoon_liqima/wenzhou_rada/out/"
files = systemfunc("ls " + diri + "*.nc")
fst = addfiles(files,"r")
;printVarSummary(fst[0])
;file = fst[0]
;;print(fst(0))
;f = addfile(file, "r")
u =fst[:]->u(:,0:5,:,:)
v =fst[:]->v(:,0:5,:,:)
;printVarSummary(u)

wsp = wind_speed(u, v)


wdir = wind_direction(u, v, 0)
;printVarSummary(wsp)
;printVarSummary(wdir)


windspeed = dim_avg_n(wsp, 1)
;wsp({time|54},{heigh|0:3},{lat|27.5},{lon|122.5})
winddirection = dim_avg_n(wdir, 1)
copy_VarCoords(wsp(:,0,:,:), winddirection)
copy_VarCoords(wsp(:,0,:,:), windspeed)
printVarSummary(windspeed)
printVarSummary(winddirection)

station_direction = winddirection({time|:},{lat|29},{lon|120.8})
station_speed = windspeed({time|:},{lat|29},{lon|120.8})
print(station_speed)

a = 5
b = 6

c =wind_speed(a, b)
print(c)
end