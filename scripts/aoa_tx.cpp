//
// Copyright 2011 Ettus Research LLC
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <fstream>
#include <csignal>
#include <complex>
#include <ctime>
#include <fftw3.h>
//#include "myUSRP.hpp"
const int decimation = 512;

namespace po = boost::program_options;

time_t seconds;
int run_time = 96;

static bool stop_signal_called = false;
void sig_int_handler(int){
	stop_signal_called = true;
}

template<typename samp_type> void send_from_file(
    uhd::usrp::multi_usrp::sptr usrp,
    const uhd::io_type_t &io_type,
    const std::string &file,
    size_t samps_per_buff,
	uhd::time_spec_t &send_time
){
    uhd::tx_metadata_t md;
    md.start_of_burst = true;
    md.end_of_burst = false;
    std::vector<samp_type> buff(samps_per_buff);
	std::vector<samp_type> buff_zeros(samps_per_buff);
    std::ifstream infile(file.c_str(), std::ifstream::binary);

	uhd::time_spec_t now = usrp->get_time_now();

	md.has_time_spec = true;
	md.time_spec = uhd::time_spec_t(0.5) + now;
	printf("send time %d %ld\n",md.time_spec.get_full_secs(),md.time_spec.get_tick_count(100e6));
	send_time = md.time_spec;

    //loop until the entire file has been read

	infile.read((char*)&buff.front(), buff.size()*sizeof(samp_type));
    size_t num_tx_samps = infile.gcount()/sizeof(samp_type);
		printf("%d\n",num_tx_samps);

	for(int i=0; i<samps_per_buff; i++){
		buff_zeros[i] = 0;	
	}

	int count=0;
	time_t start_time = time(NULL);
	time_t system_now = time(NULL);
    while(not md.end_of_burst){        
		if(system_now - start_time > run_time)
	        md.end_of_burst = true;//infile.eof();

		usrp->get_device()->send(
	            &buff.front(), num_tx_samps, md, io_type,
	            uhd::device::SEND_MODE_FULL_BUFF
	        );

		md.start_of_burst = false;
		md.has_time_spec = false;
		count++;
	system_now = time(NULL);
    }

    infile.close();
}

template<typename samp_type> void recv_to_file(
    uhd::usrp::multi_usrp::sptr usrp,
    const uhd::io_type_t &io_type,
    std::ofstream &outfile,
    size_t samps_per_buff,
	uhd::time_spec_t send_time
){
    uhd::rx_metadata_t md;
    std::vector<samp_type> buff(samps_per_buff);

	//a packet has 362 samples
	send_time = send_time + uhd::time_spec_t(1.0);
	uhd::time_spec_t front_time, end_time;
	int index;
    
    size_t num_rx_samps = usrp->get_device()->recv(
        &buff.front(), buff.size(), md, io_type,
//		uhd::device::RECV_MODE_FULL_BUFF
        uhd::device::RECV_MODE_ONE_PACKET
    );

    if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) return;
    if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE){
        throw std::runtime_error(str(boost::format(
            "Unexpected error code 0x%x"
        ) % md.error_code));
    }

	front_time = md.time_spec;
	end_time = md.time_spec + uhd::time_spec_t((double)(num_rx_samps-1)*decimation/100.0e6);

//outfile.write((const char*)&buff.front(), num_rx_samps*sizeof(samp_type));

	if((send_time-front_time).get_real_secs() < 0 && (end_time-send_time).get_real_secs() >=0 ){
		outfile.write((const char*)&buff.front(), num_rx_samps*sizeof(samp_type));
	}else{
		if( (send_time-front_time).get_real_secs() >=0 && (end_time-send_time).get_real_secs() >=0 ){
			index = (send_time-front_time).get_frac_secs()/((double)decimation/100.0e6);
			outfile.write((const char*)&buff.at(index), (num_rx_samps-index)*sizeof(samp_type));
			printf("start to save at %f with index %d, send_time %f  \n",front_time.get_real_secs(),index,send_time.get_real_secs());
			printf("timestamp %f tick %ld\n",(front_time+uhd::time_spec_t((double)index*(double)decimation/100.0e6)).get_real_secs(), (front_time+uhd::time_spec_t((double)index*(double)decimation/100.0e6)).get_tick_count(100e6));
		}
	}
    
}
/*
template<typename samp_type> void recv_to_file(
    uhd::usrp::multi_usrp::sptr usrp,
    const uhd::io_type_t &io_type,
    const std::string &file,
    size_t samps_per_buff,
	uhd::time_spec_t send_time
){
    uhd::rx_metadata_t md;
    std::vector<samp_type> buff(samps_per_buff);
    std::ofstream outfile(file.c_str(), std::ofstream::binary);

	//a packet has 362 packets
	send_time = send_time + uhd::time_spec_t(1.0);
	uhd::time_spec_t front_time, end_time;
	int index;
	bool save = false;
	time_t now = time(NULL);
    while((not stop_signal_called) and (now-seconds) <= 2){
        size_t num_rx_samps = usrp->get_device()->recv(
            &buff.front(), buff.size(), md, io_type,
            uhd::device::RECV_MODE_ONE_PACKET
        );

        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) break;
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE){
            throw std::runtime_error(str(boost::format(
                "Unexpected error code 0x%x"
            ) % md.error_code));
        }

		if(save){
			outfile.write((const char*)&buff.front(), num_rx_samps*sizeof(samp_type));
		}else{
			front_time = md.time_spec;
			end_time = md.time_spec + uhd::time_spec_t((double)(num_rx_samps-1)*128.0/100.0e6);
			if( (send_time-front_time).get_real_secs() >=0 && (end_time-send_time).get_real_secs() >=0 ){
				index = (send_time-front_time).get_frac_secs()/(128.0/100.0e6);
				save = true;
				outfile.write((const char*)&buff.at(index), (num_rx_samps-index)*sizeof(samp_type));
				printf("start to save at %f with index %d, send_time %f  \n",front_time.get_real_secs(),index,send_time.get_real_secs());
				printf("timestamp %f tick %ld\n",(front_time+uhd::time_spec_t((double)index*128.0/100.0e6)).get_real_secs(), (front_time+uhd::time_spec_t((double)index*128.0/100.0e6)).get_tick_count(100e6));
			}
		}

		now = time(NULL);
    }

    outfile.close();
}
*/

int setup_device(uhd::usrp::multi_usrp::sptr usrp, double rx_gain, double tx_gain, double freq, double rate){

    //create a usrp device
    std::cout << boost::format("Using Device: %s") % usrp->get_pp_string() << std::endl;

    usrp->set_rx_rate(rate);
    usrp->set_rx_freq(freq);
    usrp->set_rx_gain(rx_gain);
	usrp->set_tx_rate(rate);
	usrp->set_tx_freq(freq);
	usrp->set_tx_gain(tx_gain);

}

int UHD_SAFE_MAIN(int argc, char *argv[]){
    uhd::set_thread_priority_safe();

    //variables to be set by po
    std::string args, file, type, ant, subdev;
    size_t spb;
    double rate, freq, gain, bw, tx_gain, rx_gain;
	std::string run_number = "1";

	std::string ip;
	ip = "addr=192.168.10.16";
	gain = 25;
	freq = 916e6;
	rate = 100e6/decimation;

    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("file", po::value<std::string>(&file)->default_value("usrp_samples.dat"), "name of the file to read binary samples from")
        ("type", po::value<std::string>(&type)->default_value("float"), "sample type: double, float, or short")
        ("spb", po::value<size_t>(&spb)->default_value(10000), "samples per buffer")
        ("rate", po::value<double>(&rate), "rate of outgoing samples")
        ("freq", po::value<double>(&freq), "RF center frequency in Hz")
        ("gain", po::value<double>(&gain), "gain for the RF chain")
        ("ant", po::value<std::string>(&ant), "daughterboard antenna selection")
        ("subdev", po::value<std::string>(&subdev), "daughterboard subdevice specification")
        ("bw", po::value<double>(&bw), "daughterboard IF filter bandwidth in Hz")
		("run", po::value<std::string>(&run_number), "run number")
		("runtime", po::value<int>(&run_time), "run time in seconds")
		("ip", po::value<std::string>(&ip), "IP of tx antenna")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help")){
        std::cout << boost::format("UHD TX samples from file %s") % desc << std::endl;
        return ~0;
    }

	//cout << "run number" << run_number << "\n";

    //rate = 12500000;
    //rate =  7800000;
    //rate =   780000;
    //rate = 100e6/decimation;//  780000;
    //freq = 916000000;
    rx_gain = gain;//25; 
	tx_gain = gain;

	std::string tx_file, log_file;
	//tx_file = "/home/thlin/gnuradio/data_uhd/send_filtered.bin";
	tx_file = "/root/aoa/allone.dat";
	log_file = "/root/aoa/data/log_tx_";
	log_file += run_number;
	log_file += ".txt";

    FILE* logfile;
	logfile = fopen(log_file.c_str(), "w");
	fprintf(logfile, "Run: %s \n", run_number.c_str());
	fprintf(logfile, "freq: %f Hz\n",freq);
	fprintf(logfile, "rate: %f Hz\n",rate);
	fprintf(logfile, "device IP: %s \n", ip.c_str());
fclose(logfile);

    //print the help message
    if (vm.count("help")){
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        return ~0;
    }

    //create a usrp device
    std::cout << std::endl;
    //std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    uhd::usrp::multi_usrp::sptr usrp;
    usrp = uhd::usrp::multi_usrp::make(ip);
    setup_device(usrp, rx_gain, tx_gain, freq, rate);		

	usrp->set_clock_config(uhd::clock_config_t::external());	
	usrp->set_time_next_pps(uhd::time_spec_t(0.0));

	boost::this_thread::sleep(boost::posix_time::seconds(1)); //allow for some setup time 

	uhd::time_spec_t send_time;

	//send out sync message
	send_from_file<std::complex<float> >(usrp, uhd::io_type_t::COMPLEX_FLOAT32, tx_file, spb, send_time);

    //finished
    std::cout << std::endl << "Done!" << std::endl << std::endl;

    return 0;
}
