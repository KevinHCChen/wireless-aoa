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
const int decimation = 512;//128;

namespace po = boost::program_options;

time_t seconds;

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
    std::ifstream infile(file.c_str(), std::ifstream::binary);

	uhd::time_spec_t now = usrp->get_time_now();

	md.has_time_spec = true;
	md.time_spec = uhd::time_spec_t(0.5) + now;
	printf("send time %d %ld\n",md.time_spec.get_full_secs(),md.time_spec.get_tick_count(100e6));
	send_time = md.time_spec;

    //loop until the entire file has been read
    while(not md.end_of_burst){

        infile.read((char*)&buff.front(), buff.size()*sizeof(samp_type));
        size_t num_tx_samps = infile.gcount()/sizeof(samp_type);

        md.end_of_burst = infile.eof();

        usrp->get_device()->send(
            &buff.front(), num_tx_samps, md, io_type,
            uhd::device::SEND_MODE_FULL_BUFF
        );

		md.start_of_burst = false;
		md.has_time_spec = false;
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
	send_time = send_time + uhd::time_spec_t(0.1);
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
	int run_time = 96;

	std::string ip1, ip2, ip3, ip4;
	// ip1 = "addr=192.168.11.14";
	ip1 = "addr=192.168.11.14";
	ip2 = "addr=192.168.11.18";
	ip3 = "addr=192.168.11.15";
    ip4 = "addr=192.168.11.16";
    
    rate = 100e6/decimation;//  780000;
    freq = 916e6;//463e6;//916000000;
    gain = 25;

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
		("ip1", po::value<std::string>(&ip1), "IP of 1st antenna")
		("ip2", po::value<std::string>(&ip2), "IP of 2nd antenna")
		("ip3", po::value<std::string>(&ip3), "IP of 3rd antenna")
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

    //rate = 100e6/decimation;//  780000;
    //freq = 916e6;//463e6;//916000000;
    rx_gain = gain;//25; 
	tx_gain = gain;

	std::string tx_file, rx1_file, rx2_file, rx3_file, rx4_file, log_file;
	//tx_file = "/home/thlin/gnuradio/data_uhd/send_filtered.bin";
	//rx1_file = "/home/thlin/Dropbox/ofdm/ofdm_cs_ap_rx1.dat";
	//rx2_file = "/home/thlin/Dropbox/ofdm/ofdm_cs_ap_rx2.dat";
	rx1_file = "/root/aoa/data/rx1_";
	rx1_file += run_number;
	rx1_file += ".dat";
	rx2_file = "/root/aoa/data/rx2_";
	rx2_file += run_number;
	rx2_file += ".dat";
	rx3_file = "/root/aoa/data/rx3_";
	rx3_file += run_number;
	rx3_file += ".dat";
    rx4_file = "/root/aoa/data/rx4_";
    rx4_file += run_number;
    rx4_file += ".dat";
	log_file = "/root/aoa/data/log_rx_";
	log_file += run_number;
	log_file += ".txt";

    FILE* logfile;
	logfile = fopen(log_file.c_str(), "w");
	fprintf(logfile, "Run: %s \n", run_number.c_str());
	fprintf(logfile, "freq: %f Hz\n",freq);
	fprintf(logfile, "rate: %f Hz\n",rate);
	fprintf(logfile, "gain: %f\n", gain);
	fprintf(logfile, "device IP: %s %s %s\n", ip1.c_str(), ip2.c_str(), ip3.c_str());

    //print the help message
    if (vm.count("help")){
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        return ~0;
    }

    //print the help message
    if (vm.count("help")){
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        return ~0;
    }

    //create a usrp device
    std::cout << std::endl;
    //std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    uhd::usrp::multi_usrp::sptr usrp;
    usrp = uhd::usrp::multi_usrp::make(ip1);
    setup_device(usrp, rx_gain, tx_gain, freq, rate);		

    uhd::usrp::multi_usrp::sptr usrp2;
    usrp2 = uhd::usrp::multi_usrp::make(ip2);
	setup_device(usrp2, rx_gain, tx_gain, freq, rate);		

    uhd::usrp::multi_usrp::sptr usrp3;
    usrp3 = uhd::usrp::multi_usrp::make(ip3);
    setup_device(usrp3, rx_gain, tx_gain, freq, rate);		

    uhd::usrp::multi_usrp::sptr usrp4;
    usrp4 = uhd::usrp::multi_usrp::make(ip4);
    setup_device(usrp4, rx_gain, tx_gain, freq, rate);      



	usrp->set_clock_config(uhd::clock_config_t::external());	
	usrp2->set_clock_config(uhd::clock_config_t::external());
	usrp3->set_clock_config(uhd::clock_config_t::external());	
    usrp4->set_clock_config(uhd::clock_config_t::external());   


	usrp->set_time_next_pps(uhd::time_spec_t(0.0));
	usrp2->set_time_next_pps(uhd::time_spec_t(0.0));
	usrp3->set_time_next_pps(uhd::time_spec_t(0.0));
    usrp4->set_time_next_pps(uhd::time_spec_t(0.0));

	boost::this_thread::sleep(boost::posix_time::seconds(1)); //allow for some setup time 

	uhd::time_spec_t send_time;
/*
	//send out sync message
	send_from_file<std::complex<float> >(usrp, uhd::io_type_t::COMPLEX_FLOAT32, tx_file, spb, send_time);

	boost::this_thread::sleep(boost::posix_time::seconds(1));
*/
	uhd::time_spec_t now = usrp->get_time_now();
	send_time = now + uhd::time_spec_t(0.1);
	printf("time now: %f\n",now.get_real_secs());

	//receiving 
	
	//setup streaming
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = send_time;
    usrp->issue_stream_cmd(stream_cmd);
	usrp2->issue_stream_cmd(stream_cmd);
    usrp3->issue_stream_cmd(stream_cmd);
    usrp4->issue_stream_cmd(stream_cmd);

    std::signal(SIGINT, &sig_int_handler);
    std::cout << "Press Ctrl + C to stop streaming..." << std::endl;
	
	now = usrp->get_time_now();
	printf("time now: %f\n",now.get_real_secs());

	seconds = time(NULL);

	time_t system_now = time(NULL);

    std::ofstream outfile1(rx1_file.c_str(), std::ofstream::binary);
    std::ofstream outfile2(rx2_file.c_str(), std::ofstream::binary);
    std::ofstream outfile3(rx3_file.c_str(), std::ofstream::binary);
    std::ofstream outfile4(rx4_file.c_str(), std::ofstream::binary);

    while((not stop_signal_called) and (system_now-seconds) <= run_time){//48){
		recv_to_file<std::complex<float> >(usrp, uhd::io_type_t::COMPLEX_FLOAT32, outfile1, spb, send_time);
		recv_to_file<std::complex<float> >(usrp2, uhd::io_type_t::COMPLEX_FLOAT32, outfile2, spb, send_time);
		recv_to_file<std::complex<float> >(usrp3, uhd::io_type_t::COMPLEX_FLOAT32, outfile3, spb, send_time);
		recv_to_file<std::complex<float> >(usrp4, uhd::io_type_t::COMPLEX_FLOAT32, outfile4, spb, send_time);
		system_now = time(NULL);
	}
	outfile1.close();
	outfile2.close();
	outfile3.close();
	outfile4.close();

	uhd::stream_cmd_t stream_stop_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
	usrp->issue_stream_cmd(stream_stop_cmd);
	usrp2->issue_stream_cmd(stream_stop_cmd);
	usrp3->issue_stream_cmd(stream_stop_cmd);
	usrp4->issue_stream_cmd(stream_stop_cmd);

    //finished
    std::cout << std::endl << "Done!" << std::endl << std::endl;

    return 0;
}


