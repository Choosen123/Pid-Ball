#include <iostream>
#include <serialport.h>

MySerial::MySerial(string name,unsigned int baud_rate,unsigned int character_size,io_context &io)
        : port_name(name),baud_rate(baud_rate),character_size(character_size) ,sp(io,port_name)
        {

}

bool MySerial::init_and_open(){
    try{  
        sp.set_option(serial_port::baud_rate(baud_rate));
        sp.set_option(serial_port::flow_control(serial_port::flow_control::none));
        sp.set_option(serial_port::parity(serial_port::parity::none));
        sp.set_option(serial_port::stop_bits(serial_port::stop_bits::two));
        sp.set_option(serial_port::character_size(character_size));

        return true;
    }catch(const boost::system::system_error &e){
        cerr<<e.what()<<endl;
    }
    return false;
    
}

void MySerial::flush_buffer(){
        ::tcflush(sp.native_handle(), TCIOFLUSH);
}

void MySerial::my_async_read(){
    // async_read(sp,buffer(data_buffer,sizeof(data_buffer)),
            // boost::bind(&MySerial::read_callback,this,_1,_2));
        read(sp,buffer(data_buffer,sizeof(data_buffer)));
        boost::endian::little_float32_t receiveData;
        std::memcpy(&receiveData, data_buffer, sizeof(receiveData));

        angle = receiveData;
        cout<<angle<<' '<<count<<endl;
}

void MySerial::read_callback(const boost::system::error_code &ec,size_t bytes){ 
    if(!ec&&bytes>0){
        boost::endian::little_float32_t receiveData;
        std::memcpy(&receiveData, data_buffer, sizeof(receiveData));

        angle = receiveData;
        cout<<angle<<' '<<count<<endl;
    }else if(ec){
        cout<<ec.message()<<endl;
    }
}

void MySerial::my_async_write(){
    async_write(sp,buffer(&position,sizeof(position)),
    boost::bind(&MySerial::write_callback,this,_1,_2));
}

void MySerial::write_callback(const boost::system::error_code &ec, size_t bytes){
    if(!ec){
        cout<<MySerial::count++<<"send:"<<bytes<<"bytes"<<' '<<to_string(position)<<endl;
    }else{
        cerr<<"failed"<<ec.message()<<endl;
    }
}

void MySerial::close_port(){
    sp.close();
}
