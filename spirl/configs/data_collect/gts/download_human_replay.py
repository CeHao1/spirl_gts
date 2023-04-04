# core modules
from datetime import date

# project modules
from gym_gts import GTSApi

# need more changes

def download_replay(ip_address):
    with GTSApi(ip=ip_address) as gts_api:  
        gts_api.get_replays(
            player_log_dir="playerlogs",
            start_date=date(2018, 1, 1),
            end_date=date(2022, 12, 5),
            driver_ranks=["S"],
            sportsmanship_ranks=["S", "A"],
            course_code=351,  # east outer loop
            car_category="Gr.4",
            car_code = 2148,
            # car_category = 'N100'
        )


if __name__ == '___main__':
    ip = '192.168.1.5'
    download_replay(ip_address)

'''
1. enable service
curl -X POST IP:12080/ml/v1/service
curl -X POST 192.168.1.5:12080/ml/v1/service


2. search data
curl 'IP/ml/v1/discover?course_code=351&car_code=3298&after=2019-11-01&before=2019-11-15'
curl '192.168.124.34/ml/v1/discover?course_code=351&car_code=3298&after=2019-11-01&before=2019-11-15'


3. download logs
curl 'IP/ml/v1/replays/-1/log?ranking_id=xxx'

'''


'''
Instruction of replay searching.
The get_replays in the gym_gts is not working well.

1) Enable the port
curl -X POST IP:12080/ml/v1/service
curl -X POST 192.168.1.5:12080/ml/v1/service

2) search the logs
curl "IP:12080/ml/v1/discover?car_code=3298&course_code=351&dr=S&sr=S"
curl "192.168.1.5:12080/ml/v1/discover?car_code=3298&course_code=351&dr=S&sr=S"

3) download the replay

'''

def download():
    id_list = [xxx, xxx]
    with GTSApi(ip="192.168.1.5") as gts_api:  # TODO: set your PlayStation's IP-address
        for idd in id_list:
            gts_api.download_replay(
            idd=idd, 
            day='', 
            driver_rank='', 
            sportsmanship_rank='', 
            player_log_dir = './playerlogs')